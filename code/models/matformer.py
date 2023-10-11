import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm
from torch_geometric.nn import MessagePassing

import numpy as np
import math

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

from torch import Tensor


class RBFExpansion(nn.Module):
    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

class MatformerConv(MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        concat=True,
        beta=False,
        dropout= 0.0,
        edge_dim = None,
        bias = True,
        root_weight: bool = True,
    ):
        super().__init__(node_dim=0, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels,
                                   bias=bias)
            self.lin_concate = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels))
        # self.msg_layer = nn.Linear(out_channels * 3, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # self.bn = nn.BatchNorm1d(out_channels * heads)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.concat:
            self.lin_concate.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x, edge_index,
                edge_attr = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.concat:
            out = self.lin_concate(out)

        out = F.silu(self.bn(out)) # after norm and silu

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i, key_i, key_j, value_j, value_i,
                edge_attr, index, ptr,
                size_i) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
        query_i = torch.cat((query_i, query_i, query_i), dim=-1)
        key_j = torch.cat((key_i, key_j, edge_attr), dim=-1)
        alpha = (query_i * key_j) / math.sqrt(self.out_channels * 3) 
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.cat((value_i, value_j, edge_attr), dim=-1)
        out = self.lin_msg_update(out) * self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels))) 
        out = self.msg_layer(out)
        return out


class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, idx1, idx2, edges, node_features=128, edge_features=128, fc_features=128, output_features=1, heads=4, conv_layers=5):
        """Set up att modules."""
        super().__init__()

        self.edge_index = torch.vstack((idx1,idx2)).long().cuda()
        self.edge_attr = edges.cuda()
        
        self.atom_embedding = nn.Linear(
            1, node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_features,
            ),
            nn.Linear(edge_features, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features),
        )

        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=node_features, out_channels=node_features, heads=heads, edge_dim=node_features)
                for _ in range(conv_layers)
            ]
        )
        
        self.fc = nn.Sequential(
            nn.Linear(node_features, fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

       
        self.fc_out = nn.Linear(
            fc_features, output_features
        )


    def forward(self, x):
        
        bs, at = x.shape[:2]

        x = x.reshape(bs*at,-1)
        
        batch = torch.arange(bs).repeat_interleave(at).cuda()
        
        batch_edge = (torch.arange(bs).repeat_interleave(self.edge_index.shape[1])[None].repeat(2,1).cuda() * at).long()
        
        node_features = self.atom_embedding(x)
        
        edge_feat = self.edge_attr.repeat(bs, 1)
        edge_index = (self.edge_index.repeat(1,bs) + batch_edge).long()
        
        edge_features = self.rbf(edge_feat)
        
        
        for att_layer in self.att_layers:
            
            node_features = att_layer(node_features, edge_index, edge_features)
        

        # crystal-level readout
        features = scatter(node_features, batch, dim=0, reduce="mean")

        
        # features = F.softplus(features)
        features = self.fc(features)

        out = self.fc_out(features)
        
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
# building the graph:
# calculate multigraph -> draw edges to all neighbouring unit cells 
# add self edges

# look at our schnet implementation