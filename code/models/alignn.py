import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


import torch.optim as optim

from tqdm import tqdm

from torch_geometric.utils import scatter

from utils.dataloading import triplets



class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=2, keepdim=True)
            var = x.var(dim=2, unbiased=False, keepdim=True)

            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.mean(dim=0)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.mean(dim=0)

            x = (x - mean) / (var + self.eps).sqrt()
        else:
            
            x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()

        x = x * self.gamma.view(1, 1, -1) + self.beta.view(1, 1, -1)

        return x



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

class EdgeGatedGraphConv(nn.Module):
    
    def __init__(self, i, j, input_features: int, output_features: int, residual: bool = True):

        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = BatchNorm1d(output_features)
        
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = BatchNorm1d(output_features)

        self.i = i
        self.j = j
            
    def blabla(self):
        pass

    def forward(self, node_feats, edge_feats):
        e_src = self.src_gate(node_feats)[:, self.i]
        e_dst = self.dst_gate(node_feats)[:, self.j]

        y = e_src + e_dst + self.edge_gate(edge_feats)
        sigma = torch.sigmoid(y)
        bh = self.dst_update(node_feats)[:, self.j]
        m = bh*sigma

        sum_sigma_h = scatter(m, self.i, 1)

        sum_sigma = scatter(sigma, self.i, 1)

        h = sum_sigma_h/(sum_sigma+1e-6)

        x = self.src_update(node_feats) + h

        x = F.silu(x)
        y = F.suly(y)
        # x = F.silu(self.bn_nodes(x))
        # y = F.silu(self.bn_edges(y))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y

class ALIGNNConv(nn.Module):

    def __init__(self, i, j, idx_ji, idx_kj, in_features, out_features):

        super().__init__()

        self.node_update = EdgeGatedGraphConv(i,j,in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(idx_ji, idx_kj, out_features, out_features)

    def forward(self, x, y, z):

        m, z = self.edge_update(y, z)
        x, y = self.node_update(x, m)

        return x, y, z

class MLPLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        # self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        return F.silu(self.layer(x))

        # return F.silu(self.bn(self.layer(x)))

class ALIGNN(nn.Module):

    def __init__(self, idx1, idx2, X, l, embedding_features=64, triplet_input_features=40, hidden_features=256, output_features=1, mx_d=8, centers=80, a_layers=4, g_layers=4):
        super().__init__()
        i, j, idx_kj, idx_ji, angle, dist = triplets(idx2, idx1, X, l)
        self.i = i.long().cuda()
        self.j = j.long().cuda()
        self.idx_kj = idx_kj.long().cuda()
        self.idx_ji = idx_ji.long().cuda()
        self.angle = angle.float().cuda()
        self.dist = dist.float().cuda()

        self.atom_embedding = nn.Sequential(nn.Linear(1, hidden_features),
                                            # BatchNorm1d(hidden_features),
                                            nn.SiLU())
        self.edge_embedding = nn.Sequential(RBFExpansion(vmin=0, vmax=8.0, bins=centers),
                                            MLPLayer(centers, centers),
                                            MLPLayer(centers, hidden_features))
        self.angle_embedding = nn.Sequential(RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
                                             MLPLayer(triplet_input_features, centers), 
                                             MLPLayer(centers, hidden_features))

        self.alignn_layers = nn.ModuleList([ALIGNNConv(self.i, self.j, self.idx_kj, self.idx_ji, hidden_features, hidden_features) for _ in range(a_layers)])
        self.gcn_layers = nn.ModuleList([EdgeGatedGraphConv(self.i, self.j, hidden_features, hidden_features) for _ in range(g_layers)])

        self.out = nn.Linear(hidden_features, output_features)
    
    def forward(self, x):
        bs, num_nodes = x.shape[:2]
        dist = self.dist.clone()
        angle = self.angle.clone()  

        x = self.atom_embedding(x)
        y = self.edge_embedding(dist)[None].repeat(bs, 1, 1)
        z = self.angle_embedding(angle)[None].repeat(bs, 1, 1)

        for layer in self.alignn_layers:
            x, y, z = layer(x, y, z)

        for layer in self.gcn_layers:
            x, y = layer(x, y)

        h = x.mean(1)
        out = self.out(h)

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
                sites = sites.float().to('cuda')
                y = y.float().to('cuda')
                
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
                    sites = sites.float().to('cuda')
                    y = y.float().to('cuda')
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

        for idx, (sites,_,y,) in enumerate(dataloader):

            self.eval()
            
            
            
            with torch.no_grad():
                sites = sites.float().to('cuda')
                y = y.float().to('cuda')
                y_hat = self.forward(sites)
                y_hat = y_hat.reshape(y_hat.shape[0]).cpu()

                _b = y_hat.shape[0]


            y_pred[idx*b:idx*b+_b] = y_hat
            y_true[idx*b:idx*b+_b] = y

        return y_pred, y_true