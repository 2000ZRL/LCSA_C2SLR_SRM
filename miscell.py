# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:36:04 2020

@author: 14048
"""
import torch as t
import torch.nn as nn
import numpy as np
import os
import cv2
import yaml
# import arpa
# from multiprocessing import Pool
# from model import SLRModel
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
seed=10
gpu=1
t.manual_seed(seed)
t.cuda.manual_seed(seed)
t.cuda.manual_seed_all(seed)

class net1(nn.Module):
    def __init__(self, grad=True):
        super(net1, self).__init__()
        self.layer1 = nn.Linear(5,5)
        self.layer1.weight.requires_grad=grad
        self.layer2 = nn.Linear(5,5)
        self.layer3 = nn.Linear(5,5)
        self.layer4 = nn.Linear(5,5)
    
    def forward(self, x):
        iden = x
        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        return x

class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.layer = nn.Linear(5,1)
    
    def forward(self, x):
        return self.layer(x)

def lin_normalize(hmaps):
    min = hmaps.min(axis=(-2,-1), keepdims=True)
    max = hmaps.max(axis=(-2,-1), keepdims=True)
    hmaps = (hmaps - min) / (max - min + 1e-6)
    return hmaps

def gen_gaussian_hmap(coords, hmap_shape, sigma=2):
    H, W = hmap_shape
    x, y = t.meshgrid(t.arange(H), t.arange(W))
    grid = t.stack([x,y], dim=2)  #[H,H,2]
    hmap = t.exp(-((grid-coords*(H-1))**2).sum(dim=-1) / (2*sigma**2))  #[H,H]
    return hmap


# generate default config
# default = {'data': {'dataset': '2014',
#                     'aug_type': 'random_drop',
#                     'max_len': 150,
#                     'p_drop': 0.5,
#                     'resize_shape': [256,256],
#                     'crop_shape': [224,224]},
#             'model': {'name': 'lcsa',
#                       'batch_size': 2,
#                       'emb_size': 512,
#                       'vis_mod': 'vgg11',
#                       'seq_mod': 'transformer'},
#             'dcn': {'ver': 'v2',
#                     'lr_factor': 1.0},
#             'transformer':{'tf_model_size': 512,
#                            'tf_ff_size': 2048,
#                            'num_layers': 2,
#                            'num_heads': 8,
#                            'dropout': 0.1,
#                            'emb_dropout': 0.1,
#                            'pe': 'rpe_gau',
#                            'D_std_gamma': [6.3, 1.4, 2.0],
#                            'mod_D': None,
#                            'mod_src': 'Q'},
#             'comb_conv': None,
#             'qkv_context': [1, 0, 0],

#             'va': 0,
#             've': 0,
#             'sema_cons': None,
#             'att_idx_lst': [],
#             'spatial_att': None,
#             'pool_type': 'avg',
#             'cbam_pool': 'max_avg',
#             'att_sup_type': 'first',
            
#             'pose': None,
#             'pose_arg': [3, 0.5],
#             'pose_dim': 0,
#             'heatmap_num': 3,
#             'heatmap_shape': [28],
#             'heatmap_type': 'gaussian',
#             'pose_f': 1.0,

#             'from_ckpt': 0,
#             'save_dir': './results',
#             'max_num_epoch': 60,
#             'num_pretrain_epoch': 0,

#             'optimizer':{'name': 'adam',
#                          'lr': 1e-4,
#                          'betas': (0.9, 0.999),
#                          'weight_decay': 1e-4,
#                          'momentum': None},

#             'lr_scheduler': {'name': 'plateau',
#                              'decrease_factor': 0.7,
#                              'patience': 6},
            
#             'beam_size': 10,
#             'seed': 42,
#             'gpu': 1,
#             'setting': 'full',
#             'mode': 'train',
#             'test_split': 'test'
#             }

# with open('default.yml', 'w') as f:
#     yaml.dump(default, f, default_flow_style=False, sort_keys=False)


def k_means(x, center, max_steps=10):
    # x: n*d, center: k*d
    n, d = x.shape
    k = center.shape[0]
    tx = np.tile(x, (k,1))  #[n*k,d]
    for s in range(max_steps):
        print('step: ', s)
        tcenter = np.tile(center, n).reshape(n*k, d)  #[n*k,d]
        dist = ((tx-tcenter)**2).sum(axis=-1)  #[n*k]
        dist = dist.reshape(k,n).T  #[n,k]
        cluster_idx = dist.argmin(axis=-1)  #[n]
        
        # update centers
        for i in range(k):
            data = x[cluster_idx==i]
            if data.size > 0:
                center[i] = data.mean(axis=0)
        print(dist, cluster_idx, center)
    
    return center, cluster_idx


# x = np.array([[55,50], [43,50], [55,52], [43,54], [58,53], [41,47], [50,41], [50,70]], dtype=np.float32)
# center = np.array([[43,50], [55,50]], dtype=np.float32)
# cluster, center = k_means(x, center, 2)

# with open('../../data/csl-daily/csl2020ct_v1.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data.keys())
# print(data['info'][0]['name'])
# print(len(data['gloss_map']))
# info = data['info']
# df = pd.DataFrame(info)
# # df = df.sort_values(by=['name'])
# print(df[0:1])

# spl = pd.read_csv('../../data/csl-daily/split_1.txt', sep="|")
# # spl = spl.sort_values(by=['name'])
# print(spl)

# df = df.merge(spl, how='inner', on='name')
# print(df)