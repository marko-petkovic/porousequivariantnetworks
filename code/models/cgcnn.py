import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_mean, scatter_add

from models.model_utils import gaussian_basis, EmbeddingLayer

import numpy as np
from tqdm import tqdm


class ConvLayer(nn.Module):
    def __init__(self, residual=True, **kwargs):
        super(ConvLayer, self).__init__()
        self.site_len = kwargs["site_len"]
        self.bond_len = kwargs["bond_len"]
        self.residual = residual

        self.sigmoid_layer = nn.Linear(2 * self.site_len + self.bond_len, self.site_len)
        self.softmax_layer = nn.Linear(2 * self.site_len + self.bond_len, self.site_len)

    def forward(self, sites, bonds, indices1, indices2):
        sites1 = torch.index_select(sites, 1, indices1)
        sites2 = torch.index_select(sites, 1, indices2)

        vectors = torch.cat((sites1, sites2, bonds), 2)
        vectors = torch.sigmoid(self.sigmoid_layer(vectors)) * F.relu(self.softmax_layer(vectors))
        sites = sites + scatter_add(vectors, indices1, 1) if self.residual else scatter_add(vectors, indices1, 1)

        return sites


class CGCNN(nn.Module):
    def __init__(self, idx1, idx2, site_emb_size=8, edge_emb_size=8,
                 width=1, mx_d=10, mn_d=0, centers=10,
                 h1 = 24, h2 = 24, n_blocks = 6):
        super(CGCNN, self).__init__()
        
        self.idx1 = idx1.cuda()
        self.idx2 = idx2.cuda()
        
        self.in_site_len = 1
        self.in_bond_len = 1
        
        self.gaussian_kwargs = dict(max_distance=mx_d, num_centers=centers, width=width, min_distance=mn_d)
        self.site_embedding_len = site_emb_size
        self.bond_embedding_len = edge_emb_size
        
        self.h1 = h1
        self.h2 = h2
        
        
        self.site_embedding_layer = EmbeddingLayer(self.in_site_len, self.site_embedding_len)
        self.bond_embedding_layer = EmbeddingLayer(centers, self.bond_embedding_len)
        
        
        self.convs = nn.ModuleList([ConvLayer(site_len=self.site_embedding_len,    bond_len=self.bond_embedding_len) for i in range(n_blocks)])

        
        
        self.fc = nn.Sequential(nn.Linear(self.site_embedding_len, self.h1),
                                nn.ReLU(),
                                nn.Linear(self.h1, self.h2),
                                nn.ReLU(),
                                nn.Linear(self.h2, 1))
        
    def forward(self, sites, bonds):

        bonds = gaussian_basis(bonds, **self.gaussian_kwargs)

        sites = self.site_embedding_layer(sites)
        bonds = self.bond_embedding_layer(bonds)

        for conv in self.convs:
            sites = conv(sites, bonds, self.idx1, self.idx2)

        vector = sites.mean(1)
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
            for idx, (sites,bonds,y) in enumerate(trainloader):
                
                # model breaks for batch size 1
                if sites.shape[0] == 1:
                    continue
                
                self.optimizer.zero_grad()
                sites, bonds, y = sites.float().to('cuda'), bonds.float().to('cuda'), y.float().to('cuda')

                y_hat = self.forward(sites, bonds)
                y_hat = y_hat.reshape(y_hat.shape[0])
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