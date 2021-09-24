# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:11:16 2020

@author: Ronglai ZUO
flops count
"""

import torch as t
import numpy as np
from model import SetrModel, CMA
from modules.googlenet import GoogleNet
from modules.cnn import CNN
from ptflops import get_model_complexity_info
from fvcore.nn.flop_count import flop_count
from phoenix_datasets import PhoenixVideoTextDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# cma = GoogleNet()
# cnntr = CNN()

# x = t.randn(1, 3, 224, 224)

# flops_cma, _ = get_model_complexity_info(cma, (3,224,224), as_strings=True, 
#                                       print_per_layer_stat=True, verbose=True)
# print('CMA:\n', flops_cma)

# flops_cnntr, _ = get_model_complexity_info(cnntr, (3,224,224), as_strings=True, 
#                                       print_per_layer_stat=True, verbose=True)
# print('cnntr:\n', flops_cnntr)
# flop_dict, _ = flop_count(cnntr, (x,))
# print(flop_dict)

dataset = PhoenixVideoTextDataset(
                # your path to this folder, download it from official website first.
                root='../../data/phoenix2014-release/phoenix-2014-multisigner',
                split='dev',
                resize_shape=[256,256],
                crop_shape=[224,224],
                normalized_mean=[0.5372,0.5273,0.5195],
                aug_type='random_drop',
                use_random=False,
                p_drop=0.5,
                temp_scale_ratio=0.2,
                win_size=0,
                alpha=1.0
                )
dataloader = DataLoader(dataset, 
                        1,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=dataset.collate_fn,
                        drop_last=False)

cma_flop = {'googlenet': 1.6*1e9,
            'tcn': 5*512*512 + 512 + 1 + 5*512*256 + 256 + 0.5,
            'blstm': ((1024*512*2 + 1024)*4 + 512*4 + (512*512*3 + 1024)*4 + 512*4)*0.5,
            'op': 1024*1233}

cnntr_flop = {'cnn': 1.44*1e9,
              'tf_2': (512*4 + 1)*2,
              'tf_1': (1024 + (512*512+512)*3 + (512*512+512) + 512 + 1024 + \
                        (5*512*2) + 512 + 1024 + (512*2048+2048) + 2048 + \
                            (2048*512+512) + 512)*2 + 512,
              'op': 512*1233}

# model = CMA()
model = SetrModel(num_heads=8, pe='gaussian_learnable_bi', D=4.0, comb_conv='cascade',
                  qkv_context=[0,1,0], p_detach=0, vi_fea_ext='cnn')
model = model.cuda()
print(sum(t.numel(p) for p in model.parameters() if p.requires_grad) / 1e6)
model.eval()
flops = np.zeros(2, dtype=np.float64)
for i, batch_data in tqdm(enumerate(dataloader), desc='[{:s} phase, epoch {:d}]'.format('FLOP', 0)):
    with t.no_grad():
        video = t.cat(batch_data['video']).cuda()
        len_video = batch_data['len_video']
        _, _, _, _, _ = model(video, len_video)
        # len_video = float(len_video)
        # flops[0] += len_video * (cma_flop['googlenet']+cma_flop['tcn']+cma_flop['blstm']+cma_flop['op'])
        # flops[1] += len_video * (cnntr_flop['cnn']+cnntr_flop['tf_1']+cnntr_flop['op']) +\
        #             (len_video**2) * cnntr_flop['tf_2']
                    
        # flops[0] += len_video * (cma_flop['blstm'])
        # flops[1] += len_video * (cnntr_flop['tf_1']) +\
        #             (len_video**2) * cnntr_flop['tf_2']
                    
# flops /= (540*1e9)
# print(flops)
# np.savez('./phoenix_datasets/flops_cma_cnntr.npz', cma=flops[0], cnntr=flops[1])


# model = SetrModel(num_heads=8, pe='gaussian_learnable_bi', D=8.0, 
#                   comb_conv='cascade', p_detach=0, vi_fea_ext='cnn')
# model = model.cuda()
# model.eval()
# for i, batch_data in tqdm(enumerate(dataloader), desc='[{:s} phase, epoch {:d}]'.format('FLOP', 0)):
#     with t.no_grad():
#         video = t.cat(batch_data['video']).cuda()
#         len_video = batch_data['len_video'].cuda()
#         _, _, _, _, _ = model(video, len_video)      
        
        





























