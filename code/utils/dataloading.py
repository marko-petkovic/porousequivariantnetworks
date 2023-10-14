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

from torch_geometric.typing import OptTensor, SparseTensor

from itertools import product


def periodic_boundary(d):
    '''
    applies periodic boundary condition to distance d
    '''
    
    d = torch.where(d<-0.5, d+1, d)
    d = torch.where(d>0.5, d-1, d)
    
    return d

def triplets(idx1, idx2, pos, l):
    
    num_nodes=len(pos)
    
    
    row, col = idx1, idx2  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    
    pos_ji = pos[idx_j] - pos[idx_i]
    pos_kj = pos[idx_k] - pos[idx_j]
    
    # fix periodic boundary and multiply by scale
    
    pos_ji = periodic_boundary(pos_ji)*l
    pos_kj = periodic_boundary(pos_kj)*l
    
    a = (pos_ji * pos_kj).sum(dim=-1)
    b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
    angle = torch.atan2(b, a)
    
    dist = periodic_boundary(pos[col] - pos[row]).pow(2).sum(dim=-1).sqrt()
    
    return col, row, idx_kj, idx_ji, angle, dist


def get_pore(X, A_pore, l, zeo='MOR'):
    # specific for MOR
    if zeo =='MOR':
        r12 = get_area(X, A_pore[:,0]>0,l)
        r08 = get_area(np.mod(X+[.5,0,0],1), A_pore[:,2]>0,l)
        r05 = get_area(X, A_pore[:,4]>0,l)
        r04 = get_area(X, A_pore[:,13]>0,l)
        pore = np.array([r12]*2+[r08]*2+[r05]*8+[r04]*4)[:,None]
        pore2 = np.array([12]*2+[8]*2+[5]*8+[4]*4)[:,None]
        pore = np.concatenate([pore, pore2], 1)

    # specific for MFI
    elif zeo == 'MFI':

        r10t = get_area(X, A_pore[:, 5]>0, l)

        r5_1 = get_area(X, A_pore[:, 1]>0, l)
        r5_2 = get_area(X, A_pore[:, 2]>0, l)
        r6 = get_area(X, A_pore[:, 0]>0, l)

        pore = np.array([r6, r5_1, r5_2, r5_2, r5_1, r10t, r10t, r5_1, r5_1, r5_2, r5_2, r6, r10t, r10t])[:,None]
        pore2 = np.array([6, 5, 5, 5, 5, 10, 10, 5, 5, 5, 5, 6, 10, 10])[:,None]
        pore = np.concatenate([pore, pore2], 1)
        
        
        
        
    return pore
    

def get_data(l, zeo='MOR'):
    
    atoms = np.load(f'Data/{zeo}/atoms.npy').astype(int)
    hoa = np.load(f'Data/{zeo}/hoa.npy')
    
    X = np.load(f'Data/{zeo}/X.npy')

    if zeo =='MOR':
        A = pd.read_csv(f'Data/{zeo}/adj.txt', header=None, sep=' ').values[:,:-1]
    else:
        A = np.load(f'Data/{zeo}/adj.npy')
        # A_pore = pd.read_csv('data/adj_pore.csv', header=None, sep=';').values
    
    X_pore, A_pore = get_pore_X(X, l, zeo)
    
    d = get_distance_matrix(X,X,l)
    d_pore = get_distance_matrix(X, X_pore,l)
    
    pore = get_pore(X, A_pore, l, zeo)
    
    return atoms, hoa, X, A, d, X_pore, A_pore, d_pore, pore

def get_area(X, idxes, l):
    
    _X =(l*X[idxes])[:,:2]
    
    return ConvexHull(_X).volume

def get_MOR_pore(X, l):
    top_pores = np.array([
        
        [0.306, 0.926, 0.957],
        [0.415, 0.879, 0.25],
        [0.415, 0.723, 0.25],
        [0.303, 0.689, 0.956],
        [0.197, 0.811, 0.456],
        [0.085, 0.777, 0.75],
        [0.085, 0.621, 0.75],
        [0.194, 0.574, 0.457],
        [0.585, 0.879, 0.25],
        [0.585, 0.723, 0.25],
        [0.694, 0.926, 0.957],
        [0.803, 0.811, 0.456],
        [0.697, 0.689, 0.956],
        [0.915, 0.777, 0.75],
        [0.915, 0.621, 0.75],
        [0.806, 0.574, 0.457],
        [0.806, 0.426, 0.957],
        [0.915, 0.379, 0.25],
        [0.915, 0.223, 0.25],
        [0.803, 0.189, 0.956],
        [0.697, 0.311, 0.456],
        [0.585, 0.277, 0.75],
        [0.694, 0.074, 0.457],
        [0.585, 0.121, 0.75],
        [0.415, 0.277, 0.75],
        [0.415, 0.121, 0.75],
        [0.306, 0.074, 0.457],
        [0.197, 0.189, 0.956],
        [0.303, 0.311, 0.456],
        [0.194, 0.426, 0.957],
        [0.085, 0.379, 0.25],
        [0.085, 0.223, 0.25]
        
        
        
    ])
    
    _pores = np.array([
        [1,3,5],
        [3,5,13],
        [0,5,13],
        [0,4,5],
        [1,4,5],
        [1,4,14],
        [2,4,14],
        [0,2,4],
        [3,6,13],
        [0,6,13],
        [1,3,6],
        [1,7,6],
        [0,7,6],
        [1,7,14],
        [2,7,14],
        [0,2,7],
        [0,2,8],
        [2,8,15],
        [1,8,15],
        [1,8,9],
        [0,8,9],
        [0,9,12],
        [1,3,9],
        [3,9,12],
        [0,10,12],
        [3,10,12],
        [1,3,10],
        [1,10,11],
        [0,10,11],
        [0,2,11],
        [2,11,15],
        [1,11,15]
    ])

    pores_ = np.zeros((len(_pores), 16))
    
    for i in range(len(_pores)):
        pores_[i, _pores[i]] = 1

    A_pore = np.zeros((len(X), 16))
    for i in range(len(top_pores)):
        ind = np.sum(np.abs(X[:,[0,1]]-top_pores[i, [0,1]]), 1) < 0.06
    
        A_pore[ind] = pores_[i]

    X_pore = np.zeros((16, 3))
    
    for i in range(A_pore.shape[1]):

        idxes = A_pore[:,i] > 0

        pore_pts = X[idxes]

        offset = [0, 0, 0]
        if i in [14,15]:
            offset = [0.5, 0, 0]
            
        pore_pts = np.mod(pore_pts + offset, 1)
        pore_x = np.mod(np.mean(pore_pts,0) - offset, 1)

    
    
        X_pore[i] = pore_x 
    # specific for MOR
    
    X_pore[1] = [0,0,.5]
    X_pore[2] = [0,.5,.5]
    X_pore[3] = [.5,0,.5]

    return X_pore, A_pore
    

def get_MFI_pore(X, l):

    side_pore1 = np.array([
        [0.125, 0.051, 0.048],
        [0.125, 0.449, 0.048],
        [0.072, 0.036, 0.823],
        [0.072, 0.464, 0.823],
        [0.081, 0.173, 0.180],
        [0.081, 0.327, 0.180],
        [0.279, 0.054, 0.066],
        [0.279, 0.446, 0.066],
        [0.227, 0.172, 0.540],
        [0.227, 0.328, 0.540],
        [0.203, 0.431, 0.720],
        [0.203, 0.069, 0.720],
        [0.326, 0.466, 0.850],
        [0.326, 0.034, 0.850],
        [0.421, 0.429, 0.690],
        [0.421, 0.071, 0.690],
        [0.382, 0.328, 0.518],
        [0.382, 0.172, 0.518],
        [0.185, 0.123, 0.336],
        [0.185, 0.377, 0.336],
        [0.305, 0.329, 0.209],
        [0.305, 0.171, 0.209],


        [0.434, 0.379, 0.309],
        [0.434, 0.121, 0.309],
        
        [0.934, 0.379, 0.191],
        [0.934, 0.121, 0.191],
        
        [0.805, 0.329, 0.291],
        [0.805, 0.171, 0.291],
        [0.685, 0.377, 0.164],
        [0.685, 0.123, 0.164],
        [0.779, 0.054, 0.434],
        [0.779, 0.446, 0.434],
        [0.826, 0.034, 0.650],
        [0.826, 0.466, 0.650],
        [0.921, 0.429, 0.810],
        [0.921, 0.071, 0.810],
        [0.881, 0.172, 0.982],
        [0.881, 0.328, 0.982],
        [0.727, 0.172, 0.960],
        [0.727, 0.328, 0.960],
        [0.703, 0.069, 0.780],
        [0.703, 0.431, 0.780],
        [0.572, 0.464, 0.677],
        [0.572, 0.036, 0.677],
        [0.625, 0.051, 0.452],
        [0.625, 0.449, 0.452],
        [0.581, 0.327, 0.320],
        [0.581, 0.173, 0.320],
        
    ])

    side_pore2 = 1 - side_pore1
    dists = get_distance_matrix(side_pore2, X, l)
    side_pore2 = X[dists.argmin(1)]

    
    top_pores = np.array([
    [0.118, 0.828, 0.018],
    [0.079, 0.929, 0.19],
    [0.174, 0.966, 0.35],
    [0.273, 0.828, 0.04],
    [0.297, 0.931, 0.22],
    [0.221, 0.946, 0.566],
    [0.375, 0.949, 0.548],
    [0.428, 0.964, 0.323],
    [0.066, 0.879, 0.809],
    [0.195, 0.829, 0.709],
    [0.315, 0.876, 0.836],
    [0.419, 0.827, 0.68],
    [0.579, 0.929, 0.31],
    [0.566, 0.879, 0.691],
    [0.619, 0.828, 0.482],
    [0.674, 0.966, 0.15],
    [0.797, 0.931, 0.28],
    [0.928, 0.964, 0.177],
    [0.773, 0.828, 0.46],
    [0.695, 0.829, 0.791],
    [0.815, 0.876, 0.664],
    [0.919, 0.827, 0.82],
    [0.875, 0.949, 0.934],
    [0.721, 0.946, 0.934]
])


    _pores = np.array([
    [8, 10, 12],
    [7, 8, 12],
    [4, 7, 8],
    [6, 8, 10],
    [4, 6 , 8],
    [4, 5, 7],
    [1, 4, 5],
    [1, 4, 6],
    [7, 10, 12],
    [5, 7, 10],
    [5, 6, 10],
    [1, 5, 6],
    [1, 2, 6],
    [1, 3, 6],
    [1, 2, 3],
    [2, 6, 11],
    [2, 7, 11],
    [7, 11, 12],
    [2, 3, 7],
    [3, 6, 9],
    [3, 7, 9],
    [7, 9, 12],
    [2, 9, 11],
    [6, 9, 11]
])

    _pores -= 1

    pores_ = np.zeros((len(_pores), 14))

    for i in range(len(_pores)):

        pores_[i, _pores[i]] = 1
        
   
    
    A_pore = np.zeros((len(X), 14))
    for i in range(len(top_pores)):
        ind = np.sum(np.abs(X[:,[0,2]]-top_pores[i, [0,2]]), 1) < 0.06
    
        A_pore[ind] = pores_[i]

    dists = get_distance_matrix(side_pore1, X, l)
    A_pore[dists.argmin(1), 12] = 1


    dists = get_distance_matrix(side_pore2, X, l)
    A_pore[dists.argmin(1), 13] = 1
    
    X_pore = np.zeros((14, 3))
    
    for i in range(A_pore.shape[1]):

        idxes = A_pore[:,i] > 0

        pore_pts = X[idxes]

        if i in [9,10]:
            offset = [0, 0, .5]
        elif i == 11:
            offset = [.5, 0, .5]
        else:
            offset = [0, 0, 0]


        pore_pts = np.mod(pore_pts + offset, 1)

        pore_x = np.mod(np.mean(pore_pts,0) - offset, 1)

    
    
        X_pore[i] = pore_x 
    # specific for MFI
    X_pore[5] = [.5,.5,0.]
    X_pore[6] = [0,.5,.5]

    return X_pore, A_pore

def get_pore_X(X, l, zeo='MOR'):

    if zeo == 'MOR':
        X_pore, A_pore = get_MOR_pore(X, l)

    elif zeo == 'MFI':
        X_pore, A_pore = get_MFI_pore(X, l)
        
    
    # X_pore = np.zeros((12,3))
    # for i in range(A_pore.shape[1]):

    #     idxes = A_pore[:,i] > 0

    #     X_pore[i] = np.mean(X[idxes],0)
    
    # # specific for MOR
    # X_pore[1] = [0,0,.5]
    # X_pore[2] = [0,.5,.5]
    # X_pore[3] = [.5,0,.5]
    
    return X_pore, A_pore


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


def get_distance_matrix_no_pbc(X1,X2,l):
    
    d = np.zeros((X1.shape[0], X2.shape[0]))

    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):

            d[i,j] = np.sqrt(np.sum(((X1[i]-X2[j])*l)**2))

    return d

def is_self_edge(u):
    return (u == [[0,0,1],
                  [0,1,0],
                  [1,0,0],
                  [0,1,1],
                  [1,0,1],
                  [1,1,0]]).all(1).any()

def get_graph_data_mat(X, l, cutoff=5.0):
    uc = np.array(list(product([-1,0,1],[-1,0,1],[-1,0,1])))
    dm = np.zeros((X.shape[0], X.shape[0], uc.shape[0]))
    cnt = 0
    for u in uc:
        X1 = X
        X2 = X+u
        dm[:,:,cnt] = get_distance_matrix_no_pbc(X1, X2, l)
    
    
        cnt += 1


    edges, idx1, idx2 = [], [], []
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            for u in range(dm.shape[2]):
    
                if dm[i,j,u] < cutoff or (i==j and is_self_edge(uc[u])):
    
                    idx1.append(i)
                    idx2.append(j)
                    edges.append([dm[i,j,u]])

    return torch.tensor(idx1), torch.tensor(idx2), torch.tensor(edges)

def get_graph_data_ecn(X, A, l):

    # calculate fake X_o
    X_o = np.zeros((2*X.shape[0], 3))
    cnt = 0
    for i in range(A.shape[0]):
        for j in range(i,A.shape[1]):
            if A[i,j] == 1:
                D = X[j] - X[i]
                D = np.where(D>.5, D-1, D)
                D = np.where(D<-.5, D+1, D)
                # print(D.shape)
                X_o[cnt] = np.mod(X[i] + D/2, 1)
                cnt +=1

    # create super cell
    uc = list(product(range(2), range(2), range(2)))
    
    S = torch.cat([torch.tensor(X+u) for u in uc])/2
    S_o = torch.cat([torch.tensor(X_o+u) for u in uc])/2
    S_u = torch.vstack([torch.tensor(u) for u in uc]).repeat_interleave(len(X), 0)

    # calculate edges
    d_O = (S[:,None] - S_o).abs()
    d_O = np.where(d_O>0.5, d_O-1, d_O)*l*2
    d_O = (d_O**2).sum(-1)**0.5
    
    edge_index = d_O.argpartition(2, axis=0)[:2,]
    edge_index = np.concatenate([edge_index, np.flip(edge_index, 0)], 1)
    
    r, c = torch.tensor(edge_index)
    
    edge = (S[r] - S[c]).abs()
    edge = torch.where(edge>0.5, edge-1, edge)*l*2
    edge = (edge**2).sum(-1)**0.5
    unitcell = (S_u[r] - S_u[c]).abs()
    uniq = unitcell.unique(dim=0)
    comp = (unitcell[:,None] == uniq).all(-1)
    color = (comp*1).argmax(1)

    return r, c, edge.unsqueeze(-1), color
    
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



