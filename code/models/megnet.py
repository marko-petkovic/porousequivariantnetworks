import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.aggr import Set2Set

from models.model_utils import gaussian_basis, EmbeddingLayer

import numpy as np
from tqdm import tqdm

class BondUpdate(nn.Module):
    def __init__(self, site_len, bond_len, state_len, h1, h2):
        super(BondUpdate, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.state_len = state_len

        self.fc = nn.Sequential(nn.Linear(2 * self.site_len + self.bond_len + self.state_len, h1),
                                nn.ReLU(),
                                nn.Linear(h1, h2),
                                nn.ReLU(),
                                nn.Linear(h2, self.bond_len)
                               )
                                
        

    def forward(self, sites, bonds, states, indices1, indices2):
        sites1 = torch.index_select(sites, 1, indices1)
        sites2 = torch.index_select(sites, 1, indices2)
        states1 = states[:,None].repeat(1, sites1.shape[1], 1)

        vectors = torch.cat((sites1, sites2, bonds, states1), 2)

        bonds = self.fc(vectors)

        return bonds


class SiteUpdate(nn.Module):
    def __init__(self, site_len, bond_len, state_len, h1, h2):
        super(SiteUpdate, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.state_len = state_len

        self.fc = nn.Sequential(nn.Linear(site_len + bond_len + state_len, h1),
                                nn.ReLU(),
                                nn.Linear(h1, h2),
                                nn.ReLU(),
                                nn.Linear(h2, site_len),
                                nn.ReLU()
                               )
 
    def forward(self, sites, bonds, states, indices1):
        bonds_pool = self.bonds_to_site(bonds, indices1)
        states1 = states[:,None].repeat(1, sites.shape[1], 1)

        vectors = torch.cat([bonds_pool, sites, states1], 2)


        sites = self.fc(vectors)
    
        return sites

    def bonds_to_site(self, bonds, indices1):
        return scatter_mean(bonds, indices1, 1)


class StateUpdate(nn.Module):
    def __init__(self, site_len, bond_len, state_len, h1, h2):
        super(StateUpdate, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.state_len = state_len

        self.fc = nn.Sequential(nn.Linear(self.site_len + self.bond_len + self.state_len, h1),
                                nn.ReLU(),
                                nn.Linear(h1, h2),
                                nn.ReLU(),
                                nn.Linear(h2, self.state_len),
                                nn.ReLU(),
                               )

    def forward(self, sites, bonds, states):
        bonds_pool = self.bonds_to_state(bonds)
        sites_pool = self.sites_to_state(sites)

        vectors = torch.cat([bonds_pool, sites_pool, states], 1)

        states = self.fc(vectors)
        
        return states

    def bonds_to_state(self, bonds):
        return bonds.mean(1)
        
        
    def sites_to_state(self, sites):
        return sites.mean(1)

class MEGNetBlock(nn.Module):
    def __init__(
        self,
        site_len,
        bond_len,
        state_len,
        megnet_h1,
        megnet_h2,
        premegnet_h1,
        premegnet_h2,
        first_block,
    ):
        super(MEGNetBlock, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.state_len = state_len
        self.megnet_h1 = megnet_h1
        self.megnet_h2 = megnet_h2
        self.premegnet_h1 = premegnet_h1
        self.premegnet_h2 = premegnet_h2
        self.first_block = first_block

        self.bonds_fc = nn.Sequential(nn.Linear(self.bond_len, self.premegnet_h1),
                                      nn.ReLU(),
                                      nn.Linear(self.premegnet_h1, self.premegnet_h2),
                                      nn.ReLU()
                                     )
                                          
        
        self.sites_fc = nn.Sequential(nn.Linear(self.site_len, self.premegnet_h1),
                                      nn.ReLU(),
                                      nn.Linear(self.premegnet_h1, self.premegnet_h2),
                                      nn.ReLU()
                                     )
        self.states_fc = nn.Sequential(nn.Linear(self.state_len, self.premegnet_h1),
                                       nn.ReLU(),
                                       nn.Linear(self.premegnet_h1, self.premegnet_h2),
                                       nn.ReLU()
                                      )
        
        self.bondupdate = BondUpdate(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
        )
        self.siteupdate = SiteUpdate(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
        )
        self.stateupdate = StateUpdate(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
        )

    def forward(self, sites, bonds, states, indices1, indices2):
        
        
        sites1 = self.sites_fc(sites)
        bonds1 = self.bonds_fc(bonds)
        states1 = self.states_fc(states)
        
        if self.first_block:
            initial_sites, initial_bonds, initial_states = sites1, bonds1, states1

        bonds2 = self.bondupdate(sites1, bonds1, states1, indices1, indices2)
        sites2 = self.siteupdate(sites1, bonds2, states1, indices1)
        states2 = self.stateupdate(sites2, bonds2, states1)

        sites2 = sites + sites2
        bonds2 = bonds + bonds2
        states2 = states + states2

        return sites2, bonds2, states2

    

class MEGNet(nn.Module):
    def __init__(self, idx1, idx2, site_emb_size=36, edge_emb_size=36, state_emb_size=36,
                 width=0.5, mx_d=10, mn_d=0, centers=100,
                 pre_h1=64, pre_h2=32, m_h1=64, m_h2=64, post_h1=32, post_h2=16, n_blocks=3,
                 ):
        super(MEGNet, self).__init__()
        
        
        self.idx1 = idx1.cuda()
        self.idx2 = idx2.cuda()
        
        self.gaussian_kwargs = dict(max_distance=mx_d, num_centers=centers, width=width, min_distance=mn_d)
        
        self.site_embedding_len = site_emb_size
        self.bond_embedding_len = edge_emb_size
        self.state_embedding_len = state_emb_size
        
        
        self.in_bond_len = 1
        self.in_state_len = 2
        self.in_site_len = 1
        
        
        self.megnet_h1 = m_h1
        self.megnet_h2 = m_h2
        self.premegnet_h1 = pre_h1
        self.premegnet_h2 = pre_h2
        self.postmegnet_h1 = post_h1
        self.postmegnet_h2 = post_h2
        
        self.site_embedding_layer = EmbeddingLayer(self.in_site_len, self.site_embedding_len)
        self.bond_embedding_layer = EmbeddingLayer(centers, self.bond_embedding_len)
        self.state_embedding_layer = EmbeddingLayer(self.in_state_len, self.state_embedding_len)
        
        blocks = [MEGNetBlock(
            self.site_embedding_len,
            self.bond_embedding_len,
            self.state_embedding_len,
            self.megnet_h1,
            self.megnet_h2,
            self.premegnet_h1,
            self.premegnet_h2,
            True,
        )]
        
        for i in range(1,n_blocks):
            
            blocks.append(MEGNetBlock(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
            self.premegnet_h1,
            self.premegnet_h2,
            False,
        ))
        
        self.megnet_blocks = nn.ModuleList(blocks)
        
        self.sites_set2set = Set2Set(self.premegnet_h2, 3)
        self.bonds_set2set = Set2Set(self.premegnet_h2, 3)
        
        
        self.fc = nn.Sequential(nn.Linear(self.premegnet_h2 * 5, self.postmegnet_h1),
                                nn.ReLU(),
                                nn.Linear(self.postmegnet_h1, self.postmegnet_h2),
                                nn.ReLU(),
                                nn.Linear(self.postmegnet_h2, 1))
        

    def forward(self, sites, bonds, states):
        

        bs = sites.shape[0]
        
        bonds = gaussian_basis(bonds, **self.gaussian_kwargs)

        sites = self.site_embedding_layer(sites)
        bonds = self.bond_embedding_layer(bonds)
        states = self.state_embedding_layer(states)
        
        
        for megnetblock in self.megnet_blocks:
            sites, bonds, states = megnetblock(sites, bonds, states, self.idx1, self.idx2)
        
        
        graph_to_sites = torch.arange(bs).repeat_interleave(sites.shape[1]).long().cuda()#[:,None].repeat(1, sites.shape[1]).reshape(-1).long().cuda()
        graph_to_bonds = torch.arange(bs).repeat_interleave(bonds.shape[1]).long().cuda()#[:,None].repeat(1, bonds.shape[1]).reshape(-1).long().cuda()
        
        
        sites = sites.view(-1, sites.shape[-1])
        bonds = bonds.view(-1, bonds.shape[-1])
        
        sites = self.sites_set2set(sites, graph_to_sites)
        bonds = self.bonds_set2set(bonds, graph_to_bonds)
        vector = torch.cat((sites, bonds, states), 1)

        out = self.fc(vector)
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
            for idx, (sites,bonds,u,y) in enumerate(trainloader):
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                self.optimizer.zero_grad()
                sites, bonds, u, y = sites.float().to('cuda'), bonds.float().to('cuda'), u.float().to('cuda'), y.float().to('cuda')

                y_hat = self.forward(sites, bonds,u)
                y_hat = y_hat.reshape(y_hat.shape[0])
                loss = self.criterion(y_hat, y)
                if scale_loss:
                    loss /= y
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

        
            self.eval()

            for idx, (sites, bonds,u,y) in enumerate(testloader):
                
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                with torch.no_grad():
                    sites, bonds, u, y = sites.float().to('cuda'), bonds.float().to('cuda'), u.float().to('cuda'), y.float().to('cuda')
                    y_hat = self.forward(sites, bonds, u)
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

        for idx, (sites, bonds, u, y) in enumerate(dataloader):

            self.eval()
            
            
            
            with torch.no_grad():
                sites, bonds, u, y = sites.float().to('cuda'), bonds.float().to('cuda'), u.float().to('cuda'), y.float().to('cuda')
                y_hat = self.forward(sites, bonds, u)
                y_hat = y_hat.reshape(y_hat.shape[0]).cpu()

                _b = y_hat.shape[0]


            y_pred[idx*b:idx*b+_b] = y_hat
            y_true[idx*b:idx*b+_b] = y

        return y_pred, y_true
