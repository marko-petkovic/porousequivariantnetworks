import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
   
import torch_scatter as ts

import autoequiv as ae

import numpy as np

from utils.ZeoliteData import apply_symmetry, get_transform
from models.model_utils import gaussian_basis, EmbeddingLayer

from tqdm import tqdm


ACT = nn.LeakyReLU

    
class NodeUpdate(nn.Module):
    
    def __init__(self, in_features, hid_features, out_features, msg_features):
        
        super().__init__()
        
        self.layer = nn.Sequential(nn.Linear(in_features+msg_features, hid_features),
                                       ACT(),
                                       nn.Linear(hid_features, out_features),
                                       ACT())
        # TODO: fix this
        self.change_dim = False
        if in_features != out_features:
            self.change_dim = True
            self.change = nn.Linear(in_features, out_features)
        
    def forward(self, sites, messages):
        
        # concatenate node data with received message
        vectors = torch.cat([sites, messages],2)
        # process by linear layers
        vectors = self.layer(vectors)
        
        if self.change_dim:
            sites = self.change(sites)
        
        # ResNet like skip-architecture
        sites = sites + vectors
        
        return sites


class MessageUpdate(nn.Module):
    
    def __init__(self, in_features, out_features, bond_features, idx1, idx2, uc):
        
        super().__init__()
        
        
        self.idx1 = idx1.clone()
        self.idx2 = idx2.clone()
        self.uc = uc.clone()
        self.out = out_features
        
        n_layers = len(uc.unique(dim=0))
        self.n_layers = n_layers
        # layer whcih processes the message and takes care of symmetries
        self.layer1 = nn.ModuleList([nn.Sequential(
            nn.Linear(2*in_features+bond_features, out_features),
            ACT(),
            nn.Linear(out_features, out_features)) for _ in range(n_layers)])
        self.layer2 = nn.ModuleList([nn.Sequential(
            nn.Linear(2*in_features+bond_features, out_features),
            ACT(),
            nn.Linear(out_features, out_features)) for _ in range(n_layers)])
        
        self.act = ACT()
        # attention layer (determines importance of message)
        self.attention1 = nn.Sequential(nn.Linear(out_features, 1), nn.Sigmoid())
        self.attention2 = nn.Sequential(nn.Linear(out_features, 1), nn.Sigmoid())
        
        
            
    def forward(self, sites, bonds):

        sites_s = torch.index_select(sites, 1, self.idx1)
        sites_r = torch.index_select(sites, 1, self.idx2)

        vectors = torch.cat([sites_s, sites_r, bonds], 2)
        
        out1 = torch.zeros((vectors.shape[0], vectors.shape[1], self.out), device=vectors.device)
        out2 = torch.zeros((vectors.shape[0], vectors.shape[1], self.out), device=vectors.device)
        
        for i in range(self.n_layers):
            mask = self.uc == i
            out1[:,mask] = self.layer1[i](vectors[:,mask])
            out2[:,mask] = self.layer2[i](vectors[:,mask])

        out1 = self.act(out1)
        out2 = self.act(out2)
        
        lat_sites1 = self.attention1(out1) * out1
        lat_sites2 = self.attention2(out2) * out2
        
        lat_sites1 = ts.scatter_add(lat_sites1, self.idx2, 1)
        lat_sites2 = ts.scatter_add(lat_sites2, self.idx2, 1)
        # sites = self.message(sites, bonds, self.layer1, self.attention1)
        
        return lat_sites1 + lat_sites2
    

class MessagePasser(nn.Module):
    
    def __init__(self, in_features, hid_features, out_features, msg_features, bond_features, idx1, idx2, uc):
        
        super().__init__()
        
        self.node_update = NodeUpdate(in_features, hid_features, out_features, msg_features)
        self.edge_update = MessageUpdate(in_features, msg_features, bond_features, idx1, idx2, uc)
        
    def forward(self, x):
        
        sites, bonds = x
        messages = self.edge_update(sites, bonds)
        sites = self.node_update(sites, messages)
        
        return sites, bonds


class PredictionLayer(nn.Module):
    
    def __init__(self, site_pred, in_features, mlp_size, out_size, perms, pool='sum'):
        
        super().__init__()
        
        assert pool in ['sum', 'mean', 'equipool', 'fspool']
        
        self.site_pred = site_pred
        
        self.layer1 = nn.Sequential(nn.Linear(in_features, mlp_size),
                                    ACT())

        self.layer2 = nn.Sequential(nn.Linear(mlp_size, out_size))

        self.pool = pool
        
        if pool == 'sum':
            self.poolfunc = torch.sum
        elif pool == 'mean':
            self.poolfunc = torch.mean
        elif pool == 'equipool':
            self.poolfunc = nn.Sequential(ae.LinearEquiv(perms, np.zeros((perms.shape[0], 1),dtype=int), mlp_size, mlp_size),
                                          nn.LeakyReLU(),
                                         )
        elif pool == 'fspool':
            self.poolfunc = FSPool((mlp_size, perms.shape[1]))
        
            
        
    def forward(self, sites):
        
        # layer which acts locally on each node
        sites = self.layer1(sites)
        
        # if we wish to predict a global feature, perform mean aggregation of each feature
        if not self.site_pred:
            
            if self.pool in ['sum','mean']:
            
                sites = self.poolfunc(sites, 1)
            else:
                sites = sites.permute(0,2,1)
                sites = self.poolfunc(sites)
                sites = torch.squeeze(sites)
        
        sites = self.layer2(sites)
        
        return sites



class ECN(nn.Module):
    
    def __init__(self, idx1, idx2, uc,
                 site_emb_size=16, edge_emb_size=12, hid_size=[16,16,32],
                 mlp_size=32, out_size=1, site_pred=False,
                 width=1, mx_d=10, mn_d=0, centers=10, virtual=False,
                 pool='mean', set=False):
        
        super().__init__()
        
        # get permutation group
        self.site_pred = site_pred
        self.gaussian_kwargs = dict(max_distance=mx_d, num_centers=centers, width=width, min_distance=mn_d)
        
        # embedding layer for nodes and edges
        self.site_emb = EmbeddingLayer(1, site_emb_size)
        self.edge_emb = EmbeddingLayer(centers+1*virtual, edge_emb_size)
        
        # create message passing layers
        message_steps = [MessagePasser(site_emb_size, hid_size[0], hid_size[0], hid_size[0], edge_emb_size, idx1, idx2, uc)]
        message_steps.extend([MessagePasser(i, i, i, i, edge_emb_size, idx1, idx2, uc) for i in hid_size[1:]])
        
        self.message_steps = nn.Sequential(*message_steps)
        
        # create prediction layer
        self.pred = PredictionLayer(self.site_pred, hid_size[-1], mlp_size, out_size, None, pool=pool)
        
        
    def forward(self, sites, bonds):
        # gaussian embedding of distance
        bonds = gaussian_basis(bonds, **self.gaussian_kwargs)
        
        # embedding of nodes, bonds
        sites = self.site_emb(sites)
        bonds = self.edge_emb(bonds)
        
        # perform message passing steps
        sites, _ = self.message_steps((sites, bonds))
        
        # perform prediction
        pred = self.pred(sites)
        
        return pred
                    
                
        
    
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
            for idx, (sites,bonds,y) in enumerate(trainloader):
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                self.optimizer.zero_grad()
                sites, bonds, y = sites.float().to('cuda'), bonds.float().to('cuda'), y.float().to('cuda')

                y_hat = self.forward(sites, bonds)
                y_hat = y_hat.squeeze()
                #y_hat = y_hat.reshape(y_hat.shape[0])
                loss = self.criterion(y_hat, y)
                if scale_loss:
                    loss /= y
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

        
            self.eval()

            for idx, (sites, bonds,y) in enumerate(testloader):
                
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                with torch.no_grad():
                    sites, bonds, y = sites.float().to('cuda'), bonds.float().to('cuda'), y.float().to('cuda')
                    y_hat = self.forward(sites, bonds)
                    y_hat = y_hat.squeeze()
                    # y_hat = y_hat.reshape(y_hat.shape[0])
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

        for idx, (sites, bonds, y) in enumerate(dataloader):

            self.eval()
            
            
            
            with torch.no_grad():
                sites, bonds, y = sites.float().to('cuda'), bonds.float().to('cuda'), y.float().to('cuda')
                y_hat = self.forward(sites, bonds)
                y_hat = y_hat.reshape(y_hat.shape[0]).cpu()

                _b = y_hat.shape[0]


            y_pred[idx*b:idx*b+_b] = y_hat
            y_true[idx*b:idx*b+_b] = y

        return y_pred, y_true

    