import os
import sys
sys.path.append(os.getcwd())
sys.path.insert(0, "../Zeolites/code/")
sys.path.insert(0, "../Zeolites/")

import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull

from utils.ZeoliteData import get_zeolite, apply_symmetry



def get_MOR_pore(X, A_pore,l):
     # specific for MOR
    r12 = get_area(X, A_pore[:,0]>0,l)
    r08 = get_area(np.mod(X+[.5,0,0],1), A_pore[:,2]>0,l)
    r05 = get_area(X, A_pore[:,4]>0,l)
    pore = np.array([r12]*2+[r08]*2+[r05]*8)[:,None]
    pore2 = np.array([12]*2+[8]*2+[5]*8)[:,None]
    pore = np.concatenate([pore, pore2], 1)
    
    return pore
    

def get_data(l):
    
    atoms = np.load('data/atoms.npy').astype(int)
    hoa = np.load('data/hoa.npy')
    
    X = np.load('data/X.npy')
    
    A = pd.read_csv('data/adj.txt', header=None, sep=' ').values[:,:-1]
    A_pore = pd.read_csv('data/adj_pore.csv', header=None, sep=';').values
    
    X_pore = get_pore_X(X, A_pore)
    
    d = get_distance_matrix(X,X,l)
    d_pore = get_distance_matrix(X, X_pore,l)
    
    pore = get_MOR_pore(X, A_pore,l)
    
    return atoms, hoa, X, A, d, X_pore, A_pore, d_pore, pore

def get_area(X, idxes, l):
    
    _X =(l*X[idxes])[:,:2]
    
    return ConvexHull(_X).volume

def get_pore_X(X, A_pore):
    
    X_pore = np.zeros((12,3))
    for i in range(A_pore.shape[1]):

        idxes = A_pore[:,i] > 0

        X_pore[i] = np.mean(X[idxes],0)
    
    # specific for MOR
    X_pore[1] = [0,0,.5]
    X_pore[2] = [0,.5,.5]
    X_pore[3] = [.5,0,.5]
    
    return X_pore


def get_distance(a,b,l):
    
    d = np.abs(a - b)
    for i in range(3):
        if d[i] > .5:
            d[i] -= 1
    d = np.abs(d)
    d*=l # multiply by scale of unit cell
    return (d**2).sum()**.5

def get_distance_matrix(X1,X2,l):
    
    d = np.zeros((X1.shape[0], X2.shape[0]))

    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):

            d[i,j] = get_distance(X1[i], X2[j],l)

    return d


def get_graph_data(A, d):
    '''
    creates graph data
    if virtual is True, creates additional edges with distance less than max_dist
    returns:
    - edges: edge information (distance)
    - idx1: message senders
    - idx2: message receivers
    - idx2_oh: one hot encoding of idx2
    
    example: idx1[i] sends a message over edges[i] to idx2[i] 
    '''
    n_edges = int(A.sum())
        
    idx1 = torch.zeros((n_edges,), dtype = int)
    idx2 = torch.zeros((n_edges,), dtype = int)
    edges = torch.zeros((n_edges,1))
    
    
    cnt = 0
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            
            # if the edge is a covalent bond
            if A[i,j] == 1:
                idx1[cnt] = i
                idx2[cnt] = j
                edges[cnt,0] = d[i,j]
                cnt += 1
            
           
    # one hot encoding
    idx2_oh = F.one_hot(idx2)
    
    return edges, idx1, idx2, idx2_oh



