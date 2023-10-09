import os

from math import pi as PI
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing, SumAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver

from models.model_utils import EmbeddingLayer

from tqdm import tqdm

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start = 0.0,
        stop = 5.0,
        num_gaussians = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


    
    
    
def get_interaction_graph(d, cutoff):
    
    n_interactions = ((d <= cutoff)*(d>0)).sum()
    
    
    edge_index = torch.zeros((n_interactions, 2), dtype=int)
    edge_weight = torch.zeros((n_interactions,))
    
    cnt = 0
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            
            if 0 < d[i,j] <= cutoff:
                
                edge_index[cnt] = torch.tensor([i,j])
                edge_weight[cnt] = d[i,j]
                cnt += 1
                
    return edge_index, edge_weight
            
            



class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians,
                 num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x    

    
class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        
        super().__init__(aggr='add')
        
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W) -> Tensor:
        return x_j * W

    
    

class SchNet(nn.Module):
    
    def __init__(self, d, hidden_channels=128, num_filters=128, num_interactions=6,
                 num_gaussians=50, cutoff=10.0):
        
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        
        self.readout = aggr_resolver('add')
        
        edge_idx, edge_weight = get_interaction_graph(d, cutoff)
        
        self.edge_idx = edge_idx.cuda()
        self.edge_weight = edge_weight.cuda()
        
        
        self.embedding = EmbeddingLayer(1, hidden_channels)
        
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)
            
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_channels // 2, 1)
    
    def forward(self, sites):
        
        bs, at = sites.shape[:2]
        
        batch = torch.arange(bs).repeat_interleave(at).cuda()
        
        batch_edge = (torch.arange(bs).repeat_interleave(self.edge_idx.shape[0])[:,None].repeat(1,2).cuda() * at).long()
        
        #print(bs, batch_edge.min(), batch_edge.max())
        
        edge_weight = self.edge_weight.repeat(bs)
        edge_index = (self.edge_idx.repeat(bs, 1) + batch_edge).T.long()
        
        
        sites = sites.reshape(bs*at, -1)
        
        h = self.embedding(sites)
        edge_attr = self.distance_expansion(edge_weight)
        
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            
        h = self.lin1(h)
        h - self.act(h)
        h = self.lin2(h)
        
        out = self.readout(h, batch, dim=0)
        
        return out
        
    def fit(self, trainloader, testloader, epochs, crit=nn.HuberLoss, crit_kwargs={'delta':1}, opt=optim.Adam, opt_kwargs={'lr':0.001}, scale_loss=True):
        
        if scale_loss:
            crit_kwargs['reduction'] = 'none'
        
        # defines optimizer + loss criterion
        self.optimizer = opt(params=self.parameters(), **opt_kwargs)
        self.criterion = crit(**crit_kwargs)
        
        train_loss = []
        test_loss = []
        
        pbar = tqdm(range(epochs), unit='epoch', postfix='loss', position=0, leave=True)
        for e in pbar:
            self.train()
            for idx, (sites,_,y) in enumerate(trainloader):
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                self.optimizer.zero_grad()
                sites,  y = sites.float().to('cuda'), y.float().to('cuda')

                y_hat = self.forward(sites)
                y_hat = y_hat.reshape(y_hat.shape[0])
                loss = self.criterion(y_hat, y)
                if scale_loss:
                    loss /= y
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

        
            self.eval()

            for idx, (sites,_,y) in enumerate(testloader):
                
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                with torch.no_grad():
                    sites, y = sites.float().to('cuda'), y.float().to('cuda')
                    y_hat = self.forward(sites)
                    y_hat = y_hat.reshape(y_hat.shape[0])
                    loss = self.criterion(y_hat, y)
                    if scale_loss:
                        loss /= y
                    loss = loss.mean()
                    test_loss.append(loss.item())
            pbar.postfix = f'loss: {np.mean(train_loss[-len(trainloader)+1:]):.3f} test loss: {np.mean(test_loss[-len(testloader)+1:]):.3f}'

        return train_loss, test_loss
    
    def predict(self, dataloader):

        y_pred = torch.zeros((len(dataloader.dataset),))
        y_true = torch.zeros((len(dataloader.dataset),))


        b = dataloader.batch_size

        for idx, (sites,_,y) in enumerate(dataloader):

            self.eval()
            
            
            
            with torch.no_grad():
                sites, y = sites.float().to('cuda'), y.float().to('cuda')
                y_hat = self.forward(sites)
                y_hat = y_hat.reshape(y_hat.shape[0]).cpu()

                _b = y_hat.shape[0]


            y_pred[idx*b:idx*b+_b] = y_hat
            y_true[idx*b:idx*b+_b] = y

        return y_pred, y_true