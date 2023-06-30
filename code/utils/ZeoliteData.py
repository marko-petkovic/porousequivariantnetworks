import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split




zeolite_dict = {'MFI':
               {
                   'ref': 
                   
                        np.array([
                        [1,1,1],
                        [-1,1,1],
                        [1,-1,1],
                        [-1,-1,1],
                        [-1,-1,-1],
                        [1,-1,-1],
                        [-1,1,-1],
                        [1,1,-1]
                        ]),
                   
                   'tra':
                        np.array([
                        [0,0,0],
                        [.5,.5,.5],
                        [0,.5,0],
                        [.5,0,.5],
                        [0,0,0],
                        [.5,.5,.5],
                        [0,.5,0],
                        [.5,0,.5]
                        ]),
                   
                   'l': np.array([20.09,19.738,13.142]),
                   
                   'X': 
                        np.array([
                        [0.4214, 0.0711, 0.6898],
                        [0.3259, 0.0336, 0.8500],
                        [0.2792, 0.0536, 0.0655],
                        [0.1246, 0.0514, 0.0481],
                        [0.0721, 0.0360, 0.8233],
                        [0.2034, 0.0687, 0.7197],
                        [0.4195, 0.8274, 0.6805],
                        [0.3152, 0.8765, 0.8361],
                        [0.2733, 0.8278, 0.0402],
                        [0.1185, 0.8279, 0.0183],
                        [0.0657, 0.8794, 0.8087],
                        [0.1947, 0.8288, 0.7092],   
                        ]),
                   
                   'Xo':
                   
                       np.array([
                        [0.5012, 0.0699, 0.7018],
                        [0.3875, 0.0743, 0.8008],
                        [0.3995, 0.1366, 0.6251],
                        [0.3972, 0.0034, 0.6326],
                        [0.3290, 0.0385, 0.9722],
                        [0.3297, 0.9555, 0.8155],
                        [0.2571, 0.0663, 0.8106],
                        [0.2910, 0.1296, 0.1060],
                        [0.2034, 0.0455, 0.0275],
                        [0.2936, 0.0010, 0.1564],
                        [0.1075, 0.1265, 0.0883],
                        [0.0846, 0.0370, 0.9443],
                        [0.1301, 0.0781, 0.7670],
                        [0.0728, 0.9589, 0.7832],
                        [0.2201, 0.1313, 0.6456],
                        [0.4911, 0.8548, 0.7169],
                        [0.3681, 0.8311, 0.7743],
                        [0.4263, 0.7500, 0.6431],
                        [0.3217, 0.8611, 0.9561],
                        [0.2410, 0.8582, 0.7990],
                        [0.2943, 0.7500, 0.0579],
                        [0.1976, 0.8313, 0.0003],
                        [0.0948, 0.7500, 0.0205],
                        [0.0809, 0.8670, 0.9275],
                        [0.1177, 0.8370, 0.7406],
                        [0.2116, 0.7500, 0.6913]
                        ]),
                   
                   'tX':
                   np.arange(8)

               },
                
                'MOR':
                {
                    'ref':
                    # np.array([
                    #     [1,1,1],
                    #     [1,1,1],
                    #     [-1,1,1],
                    #     [-1,1,1],
                    #     [1,-1,1],
                    #     [1,-1,1],
                    #     [-1,-1,1],
                    #     [-1,-1,1],
                    #     [-1,-1,-1],
                    #     [-1,-1,-1],
                    #     [1,-1,-1],
                    #     [1,-1,-1],
                    #     [-1,1,-1],
                    #     [-1,1,-1],
                    #     [1,1,-1],
                    #     [1,1,-1]
                    # ]),
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    np.array([
                        [1,1,1],
                        [-1,-1,1],
                        [1,-1,-1],
                        [-1,1,-1],
                        [-1,-1,-1],
                        [1,1,-1],
                        [-1,1,1],
                        [1,-1,1],
                        [1,1,1],
                        [-1,-1,1],
                        [1,-1,-1],
                        [-1,1,-1],
                        [-1,-1,-1],
                        [1,1,-1],
                        [-1,1,1],
                        [1,-1,1]
                    ]),
                    
                    'tra':
                    # np.array([
                    #     [0,0,0],
                    #     [.5,.5,0],
                    #     [0,0,0],
                    #     [.5,.5,0],
                    #     [0,0,.5],
                    #     [.5,.5,.5],
                    #     [0,0,.5],
                    #     [.5,.5,.5],
                    #     [0,0,0],
                    #     [.5,.5,0],
                    #     [0,0,0],
                    #     [.5,.5,0],
                    #     [0,0,.5],
                    #     [.5,.5,.5],
                    #     [0,0,.5],
                    #     [.5,.5,.5]
                    # ]),
                               
                    
                    np.array([
                        [0,0,0],
                        [0,0,.5],
                        [0,0,0],
                        [0,0,.5],
                        [0,0,0],
                        [0,0,.5],
                        [0,0,0],
                        [0,0,.5],
                        [.5,.5,0],
                        [.5,.5,.5],
                        [.5,.5,.0],
                        [.5,.5,.5],
                        [.5,.5,0],
                        [.5,.5,.5],
                        [.5,.5,.0],
                        [.5,.5,.5]
                    ]),
                    
                    'l':np.array([18.256,20.534,7.5420]),
                    
                    'X':
                    # np.array([
                    #     [.3057, .0736, .0435],
                    #     [.3028, .3106, .0437],
                    #     [.0848, .3791, .2500],
                    #     [.0848, .2227, .2500]
                    # ]),
                    np.array([
                        [.3057, .0736, .0435],
                        [.3028, .3106, .0437],
                        [.4150, .1210, .7500],
                        [.4150, .2770, .7500]
                    ]),
                    
                    'Xo':
                    np.array([
                        [.2811, .0000, .0000],
                        [.3268, .0795, .2500],
                        [.3757, .0924, .9243],
                        [.2391, .1223, .9992],
                        [.3253, .3089, .2500],
                        [.2500, .2500, .0000],
                        [.3757, .3058, .9242],
                        [.0000, .4005, .2500],
                        [.0906, .3009, .2500],
                        [.0000, .2013, .2500]
                    ]),
                    
                    'tX': np.arange(4),
                    
                    
                }
               
               }


def get_zeolite(zeolite='MOR', sym=False):
    
    data = zeolite_dict[zeolite].copy()
    
    if not sym:
        
        syms = generate_symmetry(data['X'], data['ref'], data['tra'])  #np.mod(np.array([i[0]*data['X'] + i[1] for i in zip(data['ref'],data['tra'])]),1).reshape(-1,3)
        
        syms_o = generate_symmetry(data['Xo'],data['ref'],data['tra']) 
        #np.mod(np.array([i[0]*data['Xo'] + i[1] for i in zip(data['ref'],data['tra'])]),1).reshape(-1,3)
        
        
        _, ind = np.unique(syms, axis=0, return_index=True)
        _, ind_o = np.unique(syms_o, axis=0, return_index=True)
        
        data['X'] = syms[np.sort(ind)]
        
        
        tX = data['tX']
        
        tX = np.tile(tX, data['ref'].shape[0])
        
        data['tX'] = tX[np.sort(ind)]
            
        
    return data


def generate_symmetry(X, ref, tra):
    '''
    applies symmetry operation to set of t-atoms X to generate all coordinates
    '''
    
    return np.mod(np.array([i[0]*X + i[1] for i in zip(ref, tra)]),1).reshape(-1,3)


def apply_symmetry(X, ref, tra):
    '''
    applies symmetry operation to set of atoms X
    '''
    
    return np.mod(ref*X + tra,1)


def do_symmetry(X, at, tX, target_X):
    '''
    takes original set of atoms, coordinates and t-atoms as input
    transforms them according to new coordinates (target_X)
    '''
    #inds = np.lexsort(X.T)
    #inds_1 = np.argsort(np.lexsort(target_X.T))
    
    inds = get_transform(X, target_X)
    
    target_at = at[:,inds]#[:,inds_1]
    target_tX = tX[inds]#[inds_1]
    
    return target_at, target_tX


def get_transform(X1, X2):
    '''
    calculates the indices of the operation to go from X1 to X2
    both sets of positions are sorted lexigraphically to ensure coordinates are 1 to 1
    using the indices of the lexsort, we can calculate the transformation
    '''
    
    inds1 = np.lexsort(X1.T)
    inds2 = np.lexsort(X2.T)
    
    inds = inds1[np.argsort(inds2)]
    
    return inds


def get_sym_ops_idxes(pos, ref, tra):
    '''
    creates index operations to "go back" to the same order
    '''    
    # create idxes array of n_symmetries by n_atoms
    idxes = np.zeros((ref.shape[0], pos.shape[0]))

    for i in range(ref.shape[0]):

        new_pos = apply_symmetry(pos, ref[i], tra[i])
        idxes[i] = get_transform(new_pos, pos)

    return idxes



class ZeoliteDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]
    
    
class ZeoliteGraphDataset(Dataset):
    def __init__(self, X, bonds, y):
        self.X = X
        self.bonds = bonds
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.bonds, self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]
    
    
class ZeoliteMegDataset(Dataset):
    def __init__(self, X, bonds, y):
        self.X = X
        self.bonds = bonds
        self.u = self.get_state(X)
        self.y = y
        
    
    def __getitem__(self, idx):
        return self.X[idx], self.bonds, self.u[idx], self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]
    
    def get_state(self, X):
        
        weight = torch.where(X==1, 26.981539, 28.0855)[:,:,0].mean(1, keepdim=True)
        avg_bonds = torch.zeros_like(weight)
        avg_bonds[:,:] = 4.0
        
        u = torch.cat([weight, avg_bonds], 1)
        
       # print(u.shape)
        
        return u
        
    
    
class ZeolitePoreDataset(Dataset):
    def __init__(self, X, bonds, X_p, bonds_sp, bonds_ps, y):
        self.X = X
        self.X_p = X_p
        self.bonds = bonds
        self.bonds_sp = bonds_sp
        self.bonds_ps = bonds_ps
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.bonds, self.X_p, self.bonds_sp, self.bonds_ps, self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]
    





    
def get_data_graph(atoms, hoa, edges, bs=10, random=False, hoa_lim=None, sub_lim=None, train_geq=False, power=.75, test_size=.10, p=1):
    _X = torch.tensor(atoms).unsqueeze(-1)
    _X2 = edges
    _y = torch.tensor(hoa)
    
    # random split
    if random:
        train_idx, test_idx = train_test_split(list(range(_X.shape[0])), test_size=test_size)
    
    # split based on hoa
    elif hoa_lim is not None:
        train_idx = _y < hoa_lim
        test_idx = _y >= hoa_lim
        if train_geq:
            train_idx, test_idx = test_idx, train_idx
    
    # split based on amount of Al subs
    elif sub_lim is not None:
        train_idx = _X.sum((1,2)) < sub_lim
        test_idx = _X.sum((1,2)) >= sub_lim
        if train_geq:
            train_idx, test_idx = test_idx, train_idx
            
    
    _X_train = _X[train_idx]
    _y_train = _y[train_idx]
    
    _X_test = _X[test_idx]
    _y_test = _y[test_idx]
    
    # select a percentage of the training set
    if p < 1:
        
        np.random.seed(1234)
        
        n_train = _X_train.shape[0]
        
        tr_idx = np.random.choice(np.arange(n_train), replace=False, size=round(n_train*p))
        
        _X_train = _X_train[tr_idx]
        _y_train = _y_train[tr_idx]
    
    # create weights to perform weighted sampling 
    # this way the distribution of HoA the model trains on is more uniform
    # in practice, the trainloader is oversampling lower and higer values
    # and undersampling values in the middle
    weights = torch.zeros_like(_y_train)
    
    hist, bins = torch.histogram(_y_train)
    
    for i in range(hist.shape[0]):
        
        indices = ((bins[i] <= _y_train)*(_y_train < bins[i+1]))
        weights[indices] = 1/(hist[i]**power)
    
    
    
    wrs = WeightedRandomSampler(weights, weights.shape[0])
    
    
    trainloader = DataLoader(ZeoliteGraphDataset(_X_train, _X2, _y_train), batch_size=bs, sampler=wrs)
    testloader = DataLoader(ZeoliteGraphDataset(_X_test, _X2, _y_test), batch_size=bs, shuffle=False)

    # also create a trainloader without oversampling
    trainloader_raw = DataLoader(ZeoliteGraphDataset(_X_train, _X2, _y_train), batch_size=bs, shuffle=True)
    
    return trainloader, testloader, trainloader_raw    
    
    
def get_data_pore(atoms, hoa, edges, pore,edges_sp, edges_ps, bs=10, random=False, hoa_lim=None, sub_lim=None, train_geq=False, power=.75, test_size=.10, p=1, drop_last=False):
    _X = torch.tensor(atoms).unsqueeze(-1)
    _X2 = edges
    _y = torch.tensor(hoa)
    
    # random split
    if random:
        train_idx, test_idx = train_test_split(list(range(_X.shape[0])), test_size=test_size)
    
    # split based on hoa
    elif hoa_lim is not None:
        train_idx = _y < hoa_lim
        test_idx = _y >= hoa_lim
        if train_geq:
            train_idx, test_idx = test_idx, train_idx
    
    # split based on amount of Al subs
    elif sub_lim is not None:
        train_idx = _X.sum((1,2)) < sub_lim
        test_idx = _X.sum((1,2)) >= sub_lim
        if train_geq:
            train_idx, test_idx = test_idx, train_idx
            
    
    _X_train = _X[train_idx]
    _y_train = _y[train_idx]
    
    _X_test = _X[test_idx]
    _y_test = _y[test_idx]
    
    
    # select a percentage of the training set
    if p < 1:
        
        np.random.seed(123)
        
        n_train = _X_train.shape[0]
        
        tr_idx = np.random.choice(np.arange(n_train), replace=False, size=round(n_train*p))
        
        _X_train = _X_train[tr_idx]
        _y_train = _y_train[tr_idx]
        
    
    # create weights to perform weighted sampling 
    # this way the distribution of HoA the model trains on is more uniform
    # in practice, the trainloader is oversampling lower and higer values
    # and undersampling values in the middle
    # weights = torch.zeros_like(_y_train)
    
    # hist, bins = torch.histogram(_y_train)
    
    # for i in range(hist.shape[0]):
        
    #     indices = ((bins[i] <= _y_train)*(_y_train < bins[i+1]))
    #     weights[indices] = 1/(hist[i]**power)
    
    
    
    # wrs = WeightedRandomSampler(weights, weights.shape[0])
    
    
    #trainloader = DataLoader(ZeolitePoreDataset(_X_train, _X2, pore, edges_sp, edges_ps, _y_train), batch_size=bs, sampler=wrs)
    testloader = DataLoader(ZeolitePoreDataset(_X_test, _X2, pore, edges_sp, edges_ps, _y_test), batch_size=bs, shuffle=False, drop_last=drop_last)

    # also create a trainloader without oversampling
    trainloader_raw = DataLoader(ZeolitePoreDataset(_X_train, _X2, pore, edges_sp, edges_ps, _y_train), batch_size=bs, shuffle=True, drop_last=drop_last)
    
    return None, testloader, trainloader_raw


def get_data_megnet(atoms, hoa, edges, bs=10, random=False, hoa_lim=None, sub_lim=None, train_geq=False, power=.75, test_size=.10, p=1):
    _X = torch.tensor(atoms).unsqueeze(-1)
    _X2 = edges
    _y = torch.tensor(hoa)
    
    # random split
    if random:
        train_idx, test_idx = train_test_split(list(range(_X.shape[0])), test_size=test_size)
    
    # split based on hoa
    elif hoa_lim is not None:
        train_idx = _y < hoa_lim
        test_idx = _y >= hoa_lim
        if train_geq:
            train_idx, test_idx = test_idx, train_idx
    
    # split based on amount of Al subs
    elif sub_lim is not None:
        train_idx = _X.sum((1,2)) < sub_lim
        test_idx = _X.sum((1,2)) >= sub_lim
        if train_geq:
            train_idx, test_idx = test_idx, train_idx
            
    
    _X_train = _X[train_idx]
    _y_train = _y[train_idx]
    
    _X_test = _X[test_idx]
    _y_test = _y[test_idx]
    
    
    # select a percentage of the training set
    if p < 1:
        
        np.random.seed(123)
        
        n_train = _X_train.shape[0]
        
        tr_idx = np.random.choice(np.arange(n_train), replace=False, size=round(n_train*p))
        
        _X_train = _X_train[tr_idx]
        _y_train = _y_train[tr_idx]
        
    
    # create weights to perform weighted sampling 
    # this way the distribution of HoA the model trains on is more uniform
    # in practice, the trainloader is oversampling lower and higer values
    # and undersampling values in the middle
    weights = torch.zeros_like(_y_train)
    
    hist, bins = torch.histogram(_y_train)
    
    for i in range(hist.shape[0]):
        
        indices = ((bins[i] <= _y_train)*(_y_train < bins[i+1]))
        weights[indices] = 1/(hist[i]**power)
    
    
    
    wrs = WeightedRandomSampler(weights, weights.shape[0])
    
    
    trainloader = DataLoader(ZeoliteMegDataset(_X_train, _X2, _y_train), batch_size=bs, sampler=wrs)
    testloader = DataLoader(ZeoliteMegDataset(_X_test, _X2, _y_test), batch_size=bs, shuffle=False)

    # also create a trainloader without oversampling
    trainloader_raw = DataLoader(ZeoliteMegDataset(_X_train, _X2, _y_train), batch_size=bs, shuffle=True)
    
    return trainloader, testloader, trainloader_raw