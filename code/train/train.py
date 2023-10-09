import os
import sys
sys.path.append(os.getcwd())
sys.path.insert(0, "../porousequivariantnetworks/code/")
sys.path.insert(0, "../porousequivariantnetworks/")

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


from models.equivariant_mpnn import MPNN, MPNNPORE
from models.megnet import MEGNet
from models.cgcnn import CGCNN
from models.schnet import SchNet
from models.dimenet import DimeNetPlusPlus as DimeNet

from utils.ZeoliteData import get_zeolite, get_data_pore, get_data_graph, get_data_megnet
from utils.dataloading import get_data, get_graph_data

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-m', '--model_type', choices=['pore', 'equi','megnet','cgcnn','schnet','dime','poreset', 'equiset'], type=str)
    parser.add_argument('-z', '--zeolite', choices=['MOR', 'MFI'], type=str)
    parser.add_argument('-p', '--prop_train', type=float, default=1.0)
    parser.add_argument('-r', '--repetitions', type=int, default=1)
    parser.add_argument('-i', '--initial_repetition', type=int, default=0)
    parser.add_argument('-n', '--epochs', type=int, default=50)
    parser.add_argument('-s', '--sub_lim', type=int, default=12)
    parser.add_argument('-a', '--aggregate_pore', type=bool, default=False)
    parser.add_argument('-q', '--random_split', type=bool, default=False)

    
    args = parser.parse_args()
    
    
    #model_name = args.name
    
    for i in range(args.repetitions):
        
        print('started repetition', i)
        
        model_name = f'model_{i+1+args.initial_repetition}'
        
        data_dir = f'new_model_data/{args.zeolite}/{args.prop_train}/{args.model_type}/{model_name}/'
        if args.random_split:
            data_dir = f'new_model_data_random/{args.zeolite}/{args.prop_train}/{args.model_type}/{model_name}/'


        os.makedirs(data_dir)

        print('started!')


        data = get_zeolite(args.zeolite, sym=True)

        ref = data['ref'] # reflections
        tra = data['tra'] # translations
        l = data['l'] # scale of the unit cell

        atoms, hoa, X, A, d, X_pore, A_pore, d_pore, pore = get_data(l, args.zeolite)

        edges, idx1, idx2, idx2_oh = get_graph_data(A, d)

        if args.model_type == 'pore':

            edges_sp, idx1_sp, idx2_sp, idx2_oh_sp = get_graph_data(A_pore, d_pore)
            edges_ps, idx1_ps, idx2_ps, idx2_oh_ps = get_graph_data(A_pore.T, d_pore.T)

            mpnn = MPNNPORE(idx1.to('cuda'), idx2.to('cuda'), idx2_oh.to('cuda'), X, X_pore, ref, tra,
                            idx1_sp.to('cuda'), idx2_sp.to('cuda'), idx2_oh_sp.to('cuda'), 
                            idx1_ps.to('cuda'), idx2_ps.to('cuda'), idx2_oh_ps.to('cuda'),
                            hid_size=[8]*6, site_emb_size=8, edge_emb_size=8, mlp_size=24,
                            centers=10, mx_d=6, width=1, pool='sum', pool_pore=args.aggregate_pore).to('cuda')
            _, testloader, trainloader = get_data_pore(atoms, hoa, edges, pore, edges_sp, edges_ps, bs=32, sub_lim=args.sub_lim, p=args.prop_train, random=args.random_split)

        elif args.model_type == 'poreset':

            edges_sp, idx1_sp, idx2_sp, idx2_oh_sp = get_graph_data(A_pore, d_pore)
            edges_ps, idx1_ps, idx2_ps, idx2_oh_ps = get_graph_data(A_pore.T, d_pore.T)

            mpnn = MPNNPORE(idx1.to('cuda'), idx2.to('cuda'), idx2_oh.to('cuda'), X, X_pore, ref, tra,
                            idx1_sp.to('cuda'), idx2_sp.to('cuda'), idx2_oh_sp.to('cuda'), 
                            idx1_ps.to('cuda'), idx2_ps.to('cuda'), idx2_oh_ps.to('cuda'),
                            hid_size=[8]*6, site_emb_size=8, edge_emb_size=8, mlp_size=24,
                            centers=10, mx_d=6, width=1, pool='sum', pool_pore=args.aggregate_pore).to('cuda')
            _, testloader, trainloader = get_data_pore(atoms, hoa, edges, pore, edges_sp, edges_ps, bs=32, sub_lim=args.sub_lim, p=args.prop_train, random=args.random_split)

        
        elif args.model_type == 'equi':

            mpnn = MPNN(idx1.to('cuda'), idx2.to('cuda'), idx2_oh.to('cuda'), X, ref, tra,
                            hid_size=[8]*6, site_emb_size=8, edge_emb_size=8, mlp_size=24,
                            centers=10, mx_d=6, width=1, pool='sum').to('cuda')


            _, testloader, trainloader = get_data_graph(atoms, hoa, edges, bs=32, sub_lim=args.sub_lim, p=args.prop_train, random=args.random_split)

        elif args.model_type == 'equiset':

            mpnn = MPNN(idx1.to('cuda'), idx2.to('cuda'), idx2_oh.to('cuda'), X, ref, tra,
                            hid_size=[8]*6, site_emb_size=8, edge_emb_size=8, mlp_size=24,
                            centers=10, mx_d=6, width=1, pool='sum', set=True).to('cuda')


            _, testloader, trainloader = get_data_graph(atoms, hoa, edges, bs=32, sub_lim=args.sub_lim, p=args.prop_train, random=args.random_split)

        
        elif args.model_type == 'megnet':

            mpnn = MEGNet(idx1.to('cuda'), idx2.to('cuda')).to('cuda')


            _, testloader, trainloader = get_data_megnet(atoms, hoa, edges, bs=32, sub_lim=args.sub_lim, p=args.prop_train, random=args.random_split)

        
        elif args.model_type == 'cgcnn':

            mpnn = CGCNN(idx1.to('cuda'), idx2.to('cuda')).to('cuda')


            _, testloader, trainloader = get_data_graph(atoms, hoa, edges, bs=32, sub_lim=args.sub_lim, p=args.prop_train, random=args.random_split)
            
        
        elif args.model_type == 'schnet':
            
            mpnn = SchNet(d).to('cuda')

            
            _, testloader, trainloader = get_data_graph(atoms, hoa, edges, bs=32, sub_lim=args.sub_lim, p=args.prop_train, random=args.random_split)

        elif args.model_type == 'dime':

            mpnn = DimeNet(idx1, idx2, torch.tensor(X), torch.tensor(l)).to('cuda')

            _, testloader, trainloader = get_data_graph(atoms, hoa, edges, bs=32, sub_lim=args.sub_lim, p=args.prop_train, random=args.random_split)

        print('starting fitting!')

        lr = 0.0005 if args.model_type == 'dime' else 0.001
        
        trainloss, testloss = mpnn.fit(trainloader, testloader, args.epochs, scale_loss=False, opt=optim.AdamW,opt_kwargs={'lr':lr}, crit_kwargs={'delta':1.0})


        print('done fitting!')


        torch.save(mpnn.state_dict(), f'{data_dir}/model.pth')

        np.save(f'{data_dir}/tr_loss.npy', trainloss)

        np.save(f'{data_dir}/te_loss.npy', testloss)
   
