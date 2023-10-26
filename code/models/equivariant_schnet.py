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



def get_colors(X, ref, tra):
    colors = {}
    colors_idx = {}
    curr = 0
    
    for i in range(len(X)):
    
        added = False
        for col in colors:
            if added:
                break
            for j in range(len(ref)):
                if added:
                    break
                x = np.mod(ref[j]*X[i] + tra[j], 1)
    
                for c in colors[col]:
                    dist = np.abs(c - x).sum()
                    if dist < 0.005:
                        colors[col].append(X[i])
                        colors_idx[col].append(i)
                        added = True
                        break
                
    
        if not added:
            colors[curr] = [X[i]]
            colors_idx[curr] = [i]
            curr += 1
            
    return colors, colors_idx
    
    
def get_interaction_graph(d, A, X, ref, tra, cutoff):
    
    n_interactions = ((d <= cutoff)*(d>0)).sum()
    # n_interactions = int(A.sum())
    _, c_idx = get_colors(X, ref, tra)

    colors = torch.zeros((n_interactions,), dtype=int)
    edge_index = torch.zeros((n_interactions, 2), dtype=int)
    edge_weight = torch.zeros((n_interactions,))
    
    cnt = 0
    for i in range(A.shape[0]):

        for c in c_idx:
            if i in c_idx[c]:
                col = c
                break
        
        for j in range(A.shape[1]):
            
            if 0 < d[i,j] <= cutoff:
            # if A[i,j] == 1:  
                edge_index[cnt] = torch.tensor([i,j])
                edge_weight[cnt] = d[i,j]
                colors[cnt] = col
                cnt += 1
                
    return edge_index, edge_weight, colors
            
            
class ColoredMLP(nn.Module):

    def __init__(self, colors, num_gaussians, num_filters):
        super().__init__()

        self.colors = colors
        self.mlps = nn.ModuleList()
        for i in range(len(colors.unique())):
            mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
            )
            self.mlps.append(mlp)

        self.num_filters = num_filters

    def forward(self, edge_attr, colors):

        shape = list(edge_attr.shape[:-1]) + [self.num_filters]
        out = torch.zeros(shape, device=edge_attr.device)

        for i in range(len(self.mlps)):
            mask = colors == i
            out[mask] = self.mlps[i](edge_attr[mask])
            
        return out
            

class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians,
                 num_filters, cutoff, colors):
        super().__init__()
        self.mlp = ColoredMLP(colors, num_gaussians, num_filters)
        # self.mlp = Sequential(
        #     Linear(num_gaussians, num_filters),
        #     ShiftedSoftplus(),
        #     Linear(num_filters, num_filters),
        # )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        
        for i in range(len(self.mlp.mlps)):
            torch.nn.init.xavier_uniform_(self.mlp.mlps[i][0].weight)
            self.mlp.mlps[i][0].bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.mlp.mlps[i][2].weight)
            self.mlp.mlps[i][2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, colors):
        x = self.conv(x, edge_index, edge_weight, edge_attr, colors)
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

    def forward(self, x, edge_index, edge_weight, edge_attr, colors):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr, colors) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W) -> Tensor:
        return x_j * W

    
    

class EquiSchNet(nn.Module):
    
    def __init__(self, d, d_pore, A, A_pore, X, X_pore, ref, tra, hidden_channels=32, num_filters=32, num_interactions=8,
                 num_gaussians=50, cutoff=10.0):
        
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        
        self.readout = aggr_resolver('add')
        
        edge_idx1, edge_weight1, colors1 = get_interaction_graph(d, A, X, ref, tra, cutoff)
        edge_idx2, edge_weight2, colors2 = get_interaction_graph(d_pore, A_pore, X, ref, tra, cutoff)
        edge_idx3, edge_weight3, colors3 = get_interaction_graph(d_pore.T, A_pore.T, X_pore, ref, tra, cutoff)


        edge_idx2[:,1] += len(X)
        edge_idx3[:,0] += len(X)
        
        self.edge_idx1 = edge_idx1.cuda()
        self.edge_weight1 = edge_weight1.cuda()
        self.colors1 = colors1.cuda()
        
        self.edge_idx2 = edge_idx2.cuda()
        self.edge_weight2 = edge_weight2.cuda()
        self.colors2 = colors2.cuda()
                     
        self.edge_idx3 = edge_idx3.cuda()
        self.edge_weight3 = edge_weight3.cuda()
        self.colors3 = colors3.cuda()
        
        self.embedding = EmbeddingLayer(1, hidden_channels)
        self.embedding_p = EmbeddingLayer(2, hidden_channels)
        
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        self.interactions1 = nn.ModuleList()
        self.interactions2 = nn.ModuleList()
        self.interactions3 = nn.ModuleList()
        for _ in range(num_interactions):
            block1 = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff, self.colors1)
            block2 = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff, self.colors2)
            block3 = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff, self.colors3)
            self.interactions1.append(block1)
            self.interactions2.append(block2)
            self.interactions3.append(block3)
            
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_channels // 2, 1)
    
    def forward(self, sites, sites_p):

        m1 = sites.shape[1]
        m2 = sites_p.shape[1]

        
        h1 = self.embedding(sites)
        h2 = self.embedding_p(sites_p)

        h = torch.cat([h1,h2], 1)
        
        bs, at = h.shape[:2]

        
        mask = torch.tensor(m1*[False]+m2*[True]).repeat(bs).cuda()
        batch = torch.arange(bs).repeat_interleave(at).cuda()
        
        batch_edge1 = (torch.arange(bs).repeat_interleave(self.edge_idx1.shape[0])[:,None].repeat(1,2).cuda() * at).long()
        batch_edge2 = (torch.arange(bs).repeat_interleave(self.edge_idx2.shape[0])[:,None].repeat(1,2).cuda() * at).long()
        batch_edge3 = (torch.arange(bs).repeat_interleave(self.edge_idx3.shape[0])[:,None].repeat(1,2).cuda() * at).long()


        colors1 = self.colors1.clone()
        colors1 = colors1.repeat(bs)
        colors2 = self.colors2.clone()
        colors2 = colors2.repeat(bs)
        colors3 = self.colors3.clone()
        colors3 = colors3.repeat(bs)
        
        #print(bs, batch_edge.min(), batch_edge.max())
        
        edge_weight1 = self.edge_weight1.repeat(bs)
        edge_weight2 = self.edge_weight2.repeat(bs)
        edge_weight3 = self.edge_weight3.repeat(bs)
        
        edge_index1 = (self.edge_idx1.repeat(bs, 1) + batch_edge1).T.long()
        edge_index2 = (self.edge_idx2.repeat(bs, 1) + batch_edge2).T.long()
        edge_index3 = (self.edge_idx3.repeat(bs, 1) + batch_edge3).T.long()
        
        h = h.reshape(bs*at, -1)
        
        # h = self.embedding(sites)
        edge_attr1 = self.distance_expansion(edge_weight1)
        edge_attr2 = self.distance_expansion(edge_weight2)
        edge_attr3 = self.distance_expansion(edge_weight3)
        
        for (i1,i2,i3) in zip(self.interactions1,self.interactions2,self.interactions3):
            tt = i1(h, edge_index1, edge_weight1, edge_attr1, colors1)
            tp = i2(h, edge_index2, edge_weight2, edge_attr2, colors2)
            pt = i3(h, edge_index3, edge_weight3, edge_attr3, colors3)
            h = h + tt + tp + pt
            
        h = h[mask]
        batch = batch[mask]
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
            for idx, (sites, _, sites_p, _, _, y) in enumerate(trainloader):
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                self.optimizer.zero_grad()
                sites, sites_p, y = sites.float().to('cuda'), sites_p.float().to('cuda'), y.float().to('cuda')

                y_hat = self.forward(sites,sites_p)
                y_hat = y_hat.reshape(y_hat.shape[0])
                loss = self.criterion(y_hat, y)
                if scale_loss:
                    loss /= y
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

        
            self.eval()

            for idx, (sites, _, sites_p, _, _, y) in enumerate(testloader):
                
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                with torch.no_grad():
                    sites, sites_p, y = sites.float().to('cuda'), sites_p.float().to('cuda'), y.float().to('cuda')
                    y_hat = self.forward(sites,sites_p)
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

        for idx, (sites, _, sites_p, _, _, y) in enumerate(dataloader):

            self.eval()
            
            
            
            with torch.no_grad():
                sites, sites_p, y = sites.float().to('cuda'), sites_p.float().to('cuda'), y.float().to('cuda')
                y_hat = self.forward(sites,sites_p)
                y_hat = y_hat.reshape(y_hat.shape[0]).cpu()

                _b = y_hat.shape[0]


            y_pred[idx*b:idx*b+_b] = y_hat
            y_true[idx*b:idx*b+_b] = y

        return y_pred, y_true