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

class FSPool(nn.Module):
    
    def __init__(self, weight_shape):
        
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(weight_shape))

    def forward(self, x):

        x, _ = x.sort(dim=2, descending=True)

        return (x*self.weight).sum(2)

    
class NodeUpdatePore(nn.Module):
    
    def __init__(self, in_features, hid_features, out_features, msg_features):
        
        super().__init__()
        
        self.layer = nn.Sequential(nn.Linear(in_features+2*msg_features, hid_features),
                                       ACT(),
                                       nn.Linear(hid_features, out_features),
                                       ACT())
        # TODO: fix this
        self.change_dim = False
        if in_features != out_features:
            self.change_dim = True
            self.change = nn.Linear(in_features, out_features)
        
    def forward(self, sites, messages1, messages2):
        
        # concatenate node data with received message
        vectors = torch.cat([sites, messages1, messages2],2)
        # process by linear layers
        vectors = self.layer(vectors)
        
        if self.change_dim:
            sites = self.change(sites)
        
        # ResNet like skip-architecture
        # TODO: add weight to sites/vectors
        sites = sites + vectors
        
        return sites
    
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


class MessageUpdatePore(nn.Module):
    
    def __init__(self, in_features, out_features, bond_features, idx1, idx2, idx2_oh, perms1, perms2):
        
        super().__init__()
        
        self.idx1 = idx1.clone()
        self.idx2 = idx2.clone()
        self.idx2_oh = idx2_oh.clone()
        self.out = out_features
        
        
        # layer whcih processes the message and takes care of symmetries
        self.layer1 = nn.Sequential(ae.LinearEquiv(perms1, perms2, 2*in_features + bond_features, out_features, bias=True), ACT())
        
        # attention layer (determines importance of message)
        self.attention1 = nn.Sequential(nn.Linear(out_features, 1), nn.Sigmoid())
        
        

            
    def forward(self, sites1, sites2, bonds):
        
        sites = self.message(sites1, sites2, bonds, self.layer1, self.attention1)
        
            
        return sites
        
        
    
    def message(self, sites, sites_p, bonds, layer, att_layer):
        # select senders and receivers
        sites_s = torch.index_select(sites, 1, self.idx1)
        sites_r = torch.index_select(sites_p, 1, self.idx2)
        
        # concatenates senders, receivers and bond information
        vectors = torch.cat([sites_s, sites_r, bonds], 2)
     
        # "reshape" the data such that it looks like the adjacency matrix
        vectors_cells = torch.einsum("bij,ik->bijk", vectors, self.idx2_oh)#.permute(0,1,3,2)
                
        # process by "symmetry layer" 
        vectors_cells = layer(vectors_cells)
        
        # get indices of sending nodes
        idx2_cells = self.idx2[None, :, None, None].repeat(sites_p.shape[0], 1, self.out, 1)
        
        # collect the message from these nodes
        lat_sites = torch.gather(vectors_cells, 3, idx2_cells, sparse_grad=False).squeeze()
        
        
        # apply attention
        lat_sites = att_layer(lat_sites) * lat_sites
        
        # sum messages
        lat_sites = ts.scatter_add(lat_sites, self.idx2, 1)
        
        return lat_sites
        

class MessageUpdate(nn.Module):
    
    def __init__(self, in_features, out_features, bond_features, idx1, idx2, idx2_oh, perms):
        
        super().__init__()
        
        
        self.perms = perms
        self.idx1 = idx1.clone()
        self.idx2 = idx2.clone()
        self.idx2_oh = idx2_oh.clone()
        self.out = out_features
        
        # layer whcih processes the message and takes care of symmetries
        self.layer1 = nn.Sequential(ae.LinearEquiv(perms, perms, 2*in_features + bond_features, out_features, bias=True), ACT())
        
        # attention layer (determines importance of message)
        self.attention1 = nn.Sequential(nn.Linear(out_features, 1), nn.Sigmoid())
        
    
            
    def forward(self, sites, bonds):
        
        sites = self.message(sites, bonds, self.layer1, self.attention1)
        
        return sites
        
        
    
    def message(self, sites, bonds, layer, att_layer):
        # select senders and receivers
        sites_s = torch.index_select(sites, 1, self.idx1)
        sites_r = torch.index_select(sites, 1, self.idx2)
        
        # concatenates senders, receivers and bond information
        vectors = torch.cat([sites_s, sites_r, bonds], 2)
        
        # "reshape" the data such that it looks like the adjacency matrix
        vectors_cells = torch.einsum("bij,ik->bijk", vectors, self.idx2_oh)
        
        # process by "symmetry layer" 
        vectors_cells = layer(vectors_cells)
        
        # get indices of sending nodes
        idx2_cells = self.idx2[None, :, None, None].repeat(sites.shape[0], 1, self.out, 1)
        
        # collect the message from these nodes
        lat_sites = torch.gather(vectors_cells, 3, idx2_cells, sparse_grad=False).squeeze()
        
        # apply attention
        lat_sites = att_layer(lat_sites) * lat_sites
        
        # sum messages
        lat_sites = ts.scatter_add(lat_sites, self.idx2, 1)
        
        return lat_sites
    

class MessagePasser(nn.Module):
    
    def __init__(self, in_features, hid_features, out_features, msg_features, bond_features, idx1, idx2, idx2_oh, perms):
        
        super().__init__()
        
        self.node_update = NodeUpdate(in_features, hid_features, out_features, msg_features)
        self.edge_update = MessageUpdate(in_features, msg_features, bond_features, idx1, idx2, idx2_oh, perms)
        
    def forward(self, x):
        
        sites, bonds = x
        messages = self.edge_update(sites, bonds)
        sites = self.node_update(sites, messages)
        
        return sites, bonds


class MessagePasserPore(nn.Module):
    
    def __init__(self, in_features, hid_features, out_features, msg_features, bond_features,
                 idx1, idx2, idx2_oh, perms, perms_p, 
                 idx1_sp, idx2_sp, idx2_oh_sp,
                 idx1_ps, idx2_ps, idx2_oh_ps):
        
        super().__init__()
        
        self.node_update = NodeUpdatePore(in_features, hid_features, out_features, msg_features)
        self.node_update_p = NodeUpdate(in_features, hid_features, out_features, msg_features)
        
        self.edge_update = MessageUpdate(in_features, msg_features, bond_features, idx1, idx2, idx2_oh, perms)
        self.edge_update_sp = MessageUpdatePore(in_features, msg_features, bond_features, idx1_sp, idx2_sp, idx2_oh_sp, perms_p, perms_p)
        self.edge_update_ps = MessageUpdatePore(in_features, msg_features, bond_features, idx1_ps, idx2_ps, idx2_oh_ps, perms, perms)
        
        
        
        
    def forward(self, x):
        
        sites, bonds, sites_p, bonds_sp, bonds_ps = x
        messages = self.edge_update(sites, bonds)
        
        messages_ps = self.edge_update_ps(sites_p, sites, bonds_ps)
        messages_sp = self.edge_update_sp(sites, sites_p, bonds_sp)
        
        sites = self.node_update(sites, messages, messages_ps)
        sites_p = self.node_update_p(sites_p, messages_sp)
        
        return sites, bonds, sites_p, bonds_sp, bonds_ps



class PredictionLayer(nn.Module):
    
    def __init__(self, site_pred, in_features, mlp_size, out_size, perms, pool='sum'):
        
        super().__init__()
        
        assert pool in ['sum', 'mean', 'equipool', 'fspool']
        
        self.site_pred = site_pred
        
        self.layer1 = nn.Sequential(nn.Linear(in_features, mlp_size),
                                    ACT())

        self.layer2 = nn.Sequential(nn.Linear(mlp_size, mlp_size),
                                    ACT(),
                                    nn.Linear(mlp_size, mlp_size),
                                    ACT(),
                                    nn.Linear(mlp_size, out_size))

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



class MPNN(nn.Module):
    
    def __init__(self, idx1, idx2, idx2_oh, X, ref, tra,
                 site_emb_size=16, edge_emb_size=12, hid_size=[16,16,32],
                 mlp_size=32, out_size=1, site_pred=False,
                 width=1, mx_d=10, mn_d=0, centers=10, virtual=False,
                 pool='none'):
        
        super().__init__()
        
        # get permutation group
        self.perms = self.get_perms(X, ref, tra)
        self.site_pred = site_pred
        self.gaussian_kwargs = dict(max_distance=mx_d, num_centers=centers, width=width, min_distance=mn_d)
        
        # embedding layer for nodes and edges
        self.site_emb = EmbeddingLayer(1, site_emb_size)
        self.edge_emb = EmbeddingLayer(centers+1*virtual, edge_emb_size)
        
        # create message passing layers
        message_steps = [MessagePasser(site_emb_size, hid_size[0], hid_size[0], hid_size[0], edge_emb_size, idx1, idx2, idx2_oh, self.perms)]
        message_steps.extend([MessagePasser(i, i, i, i, edge_emb_size, idx1, idx2, idx2_oh, self.perms) for i in hid_size[1:]])
        
        self.message_steps = nn.Sequential(*message_steps)
        
        # create prediction layer
        self.pred = PredictionLayer(self.site_pred, hid_size[-1], mlp_size, out_size, self.perms, pool=pool)
        
        
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
        
        
        
        
    def get_perms(self, X, ref, tra):    
        permutations = np.zeros((ref.shape[0], X.shape[0]),int)
        for i in range(ref.shape[0]):
            old_X = X.copy()
            new_X = apply_symmetry(old_X, ref[i], tra[i])
            permutations[i] = get_transform(old_X, new_X) 
        return permutations
    
                
                
        
    
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

    
class MPNNPORE(nn.Module):
    
    def __init__(self, idx1, idx2, idx2_oh, X, X_p, ref, tra,
                 idx1_sp, idx2_sp, idx2_oh_sp, 
                 idx1_ps, idx2_ps, idx2_oh_ps, 
                 site_emb_size=16, edge_emb_size=12, hid_size=[16,16,32],
                 mlp_size=32, out_size=1, site_pred=False,
                 width=1, mx_d=10, mn_d=0, centers=10,
                 pool='none', add_p=False):
        
        super().__init__()

        self.add_p = add_p
                     
        # get permutation group
        self.perms = self.get_perms(X, ref, tra)
        self.perms_p = self.get_perms(X_p, ref, tra)
        self.site_pred = site_pred
        
        self.gaussian_kwargs = dict(max_distance=mx_d, num_centers=centers, width=width, min_distance=mn_d)
        
        # embedding layer for nodes and edges
        self.site_emb = EmbeddingLayer(1+1*add_p, site_emb_size)
        self.edge_emb = EmbeddingLayer(centers, edge_emb_size)
        
        self.site_emb_p = EmbeddingLayer(2+1*add_p, site_emb_size)
        self.edge_emb_p = EmbeddingLayer(centers, edge_emb_size)
        
        # create message passing layers
        message_steps = [MessagePasserPore(site_emb_size, hid_size[0], hid_size[0], hid_size[0], edge_emb_size, idx1, idx2, idx2_oh, self.perms, self.perms_p, idx1_sp, idx2_sp, idx2_oh_sp,
                 idx1_ps, idx2_ps, idx2_oh_ps)]
        message_steps.extend([MessagePasserPore(i, i, i, i, edge_emb_size, idx1, idx2, idx2_oh, self.perms, self.perms_p, idx1_sp, idx2_sp, idx2_oh_sp,
                 idx1_ps, idx2_ps, idx2_oh_ps) for i in hid_size[1:]])
        
        self.message_steps = nn.Sequential(*message_steps)
        
        # create prediction layer
        self.pred = PredictionLayer(self.site_pred, hid_size[-1], mlp_size, out_size, self.perms, pool=pool)
        
        
    def forward(self, sites, bonds, sites_p, bonds_sp, bonds_ps, p=None):

        if p is not None:

            sites = torch.cat([sites, p], -1)
            sites_p = torch.cat([sites_p, p], -1)
        
        # gaussian embedding of distance
        bonds = gaussian_basis(bonds, **self.gaussian_kwargs)
        bonds_sp = gaussian_basis(bonds_sp, **self.gaussian_kwargs)
        bonds_ps = gaussian_basis(bonds_ps, **self.gaussian_kwargs)
        
        # embedding of nodes, bonds
        sites = self.site_emb(sites)
        bonds = self.edge_emb(bonds)
        
        sites_p = self.site_emb_p(sites_p)
        bonds_sp = self.edge_emb_p(bonds_sp)
        bonds_ps = self.edge_emb_p(bonds_ps)
        
        
        # perform message passing steps
        sites, _, sites_p, _, _ = self.message_steps((sites, bonds, sites_p, bonds_sp, bonds_ps))
        
        # perform prediction
        pred = self.pred(sites)
        
        return pred
        
        
        
        
    def get_perms(self, X, ref, tra):    
        permutations = np.zeros((ref.shape[0], X.shape[0]),int)
        for i in range(ref.shape[0]):
            old_X = X.copy()
            new_X = apply_symmetry(old_X, ref[i], tra[i])
            permutations[i] = get_transform(old_X, new_X) 
        return permutations
    
                
                
        
    
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
            for idx, (sites, bonds, sites_p, bonds_sp, bonds_ps, y) in enumerate(trainloader):
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                    
                self.optimizer.zero_grad()
                sites, bonds, sites_p, bonds_sp, bonds_ps, y = sites.float().to('cuda'), bonds.float().to('cuda'), sites_p.float().to('cuda'), bonds_sp.float().to('cuda'), bonds_ps.float().to('cuda'), y.float().to('cuda')
                y_hat = self.forward(sites, bonds, sites_p, bonds_sp, bonds_ps)
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

            for idx, (sites, bonds, sites_p, bonds_sp, bonds_ps, y) in enumerate(testloader):
                
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                with torch.no_grad():
                    sites, bonds, sites_p, bonds_sp, bonds_ps, y = sites.float().to('cuda'), bonds.float().to('cuda'), sites_p.float().to('cuda'), bonds_sp.float().to('cuda'), bonds_ps.float().to('cuda'), y.float().to('cuda')
                    y_hat = self.forward(sites, bonds, sites_p, bonds_sp, bonds_ps)
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

        for idx, (sites, bonds, sites_p, bonds_sp, bonds_ps, y) in enumerate(dataloader):

            self.eval()

            with torch.no_grad():
                sites, bonds, sites_p, bonds_sp, bonds_ps, y = sites.float().to('cuda'), bonds.float().to('cuda'), sites_p.float().to('cuda'), bonds_sp.float().to('cuda'), bonds_ps.float().to('cuda'), y.float().to('cuda')
                y_hat = self.forward(sites, bonds, sites_p, bonds_sp, bonds_ps)
                y_hat = y_hat.reshape(y_hat.shape[0]).cpu()

                _b = y_hat.shape[0]


            y_pred[idx*b:idx*b+_b] = y_hat
            y_true[idx*b:idx*b+_b] = y

        return y_pred, y_true
