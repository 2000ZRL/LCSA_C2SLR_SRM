# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:45:31 2020

@author: Ronglai ZUO
TCN
"""
import torch as t
import torch.nn as nn
from tfmer.embeddings import MaskedNorm
from utils.utils import create_mask


class TCN_block(nn.Module):
    def __init__(self, 
                 inchannels, 
                 outchannels, 
                 kernel_size, 
                 stride=1, 
                 groups=1,
                 norm_type='batch', 
                 use_pool=True,
                 use_glu=False, 
                 act_fun='relu'
                 ):
        super(TCN_block, self).__init__()
        self.use_pool = use_pool
        self.use_glu = use_glu
        if use_glu:
            outchannels = 2 * outchannels
        
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            raise ValueError('We only support kernel_size = 3 or 5')
        
        self.conv = nn.Conv1d(inchannels, outchannels, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm = MaskedNorm(norm_type=norm_type,
                               num_groups=1,
                               num_features=outchannels)
        if act_fun == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_fun == 'swish':
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError('We only support relu and swish act fun.')
        
        if self.use_pool:
            self.max_pooling = nn.MaxPool1d(kernel_size=2)  #To make shape match with shortcut
        
    def forward(self, x, len_video):
        '''
        Input: [B, C, max_T]
        mask: True for padding values!
        '''
        mask = create_mask(len_video)
        x = self.conv(x)
        x = self.norm(x, mask)
        
        if not self.use_glu:
            #glu and se can provide non-linearty
            x = self.act(x)
        else:
            #gated linear unit
            x, gate = x.split(x.shape[1]//2, dim=1)
            x = x * t.sigmoid(gate)
        
        #masked conv, ref: HIGH FIDELITY SPEECH SYNTHESIS WITH ADVERSARIAL NETWORKS
        # x = x.mul(mask<=0)
        if self.use_pool:
            x = self.max_pooling(x)
            len_video = len_video//2

        return x, len_video
