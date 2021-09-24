# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 19:10:52 2020

@author: Ronglai ZUO
Script for Aligner and SoftDTW
Ref: END-TO-END ADVERSARIAL TEXT-TO-SPEECH  ICLR 2021. See appendix
"""

import torch as t
import torch.nn as nn
import numpy as np


class Aligner(nn.Module):
    #To generate aligned features using length predictor
    def __init__(self, emb_size=512):
        super(Aligner, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        #We don't need masked conv because kernel_size=1
        self.conv1 = nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=emb_size, out_channels=1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    
    def forward(self, x, len_fea, out_seq_len=80, sigma2=10.0, rec=True):
        '''
        Parameters
        ----------
        x : cuda Tensor, gls
            [B,C,MAX_T].
        len_fea : Tensor
            [B]
        len_fea : Tensor
            [B]
        rec : bool
            True means this aligner is for recognition, othervise, it is for generation
        Returns
        -------
        None.
        '''
        b_size = len_fea.shape[0]
        T = x.shape[-1]
        unaligned_features = x
        
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if rec:
            x = self.sigmoid(x)  #the ratio of this frame to the gloss
        else:
            x = self.relu(x)  #[B,1,MAX_T]
        
        x = t.squeeze(x, dim=1)  #[B,MAX_T], gls_lengths
        gls_ends = t.cumsum(x, dim=1)
        gls_centers = gls_ends - (x/2)  #[B,MAX_T]
        
        aligned_lengths = gls_ends.view(-1)
        index = t.arange(0, b_size*T, T).cuda() + len_fea - 1
        aligned_lengths = t.index_select(aligned_lengths, dim=0, index=index)  #[B]
        del index
        # aligned_lengths = [end[length-1] for end, length in zip(gls_ends, len_fea)]  #[B]
        
        if not rec:
            out_pos = t.arange(out_seq_len, dtype=t.float).expand(b_size, out_seq_len)[:,:,None].cuda()  #[B,seq,1]
        else:
            out_pos = (0.5+t.arange(out_seq_len, dtype=t.float)).expand(b_size, out_seq_len)[:,:,None].cuda()  #[B,seq,1]
        weights = gls_centers[:, None, :] - out_pos  #[B,seq,MAX_T]
        weights = -(weights**2 / sigma2)
        
        mask = t.zeros(b_size, 1, max(len_fea)).cuda()
        for i, l in enumerate(len_fea):
            mask[i, 0, l:] = 1.0
        
        weights -= 1e9 * mask
        weights = self.softmax(weights)  #[B,seq,MAX_T]
        # print(weights.shape)
        aligned_features = t.matmul(weights, unaligned_features.transpose(1,2))  #[B,seq,C]
        
        return aligned_features, aligned_lengths


def soft_minimum(values, temp):
    '''
    Parameters
    ----------
    values : Tensor
        [3,size_a].
    temp : float
        DESCRIPTION.

    Returns
    -------
    Tensor with shape [size_a]
    '''
    return -temp * t.log(t.sum(t.exp(-values/temp), dim=0))
    

def skew_matrix(x):
    #skew a matrix so that the diagonals become the rows
    height, width = x.shape
    y = t.zeros(height+width-1, width)
    for i in range(height+width-1):
        for j in range(width):
            y[i,j] = x[np.clip(i-j, 0, height-1), j]
    return y


class Soft_DTW(nn.Module):
    def __init__(self, warp_penalty=1.0, temp=0.01, dist='L1', reduction='mean'):
        super(Soft_DTW, self).__init__()
        self.warp_penalty = warp_penalty
        self.temp = temp
        self.dist = dist
        self.reduction = reduction
        
    def forward(self, fea_a, len_a, fea_b, len_b):
        '''
        Parameters
        ----------
        fea_a : Tensor
            [B,T1,C].
        fea_b : Tensor
            [B,T2,C].
        len_a : Tensor
            [B]
        len_b : Tensor
            [B]
        
        Returns
        -------
        loss, i.e., minimum path cost
        '''
        if self.dist in ['L1', 'L2']:
            diff = t.abs(fea_a[:,None,:,:] - fea_b[:,:,None,:])  #[B,T2,T1,C]
            if self.dist == 'L2':
                diff = diff**2
            cost = t.mean(diff, dim=-1)  #[B,T2,T1]
        
        elif self.dist == 'cos':
            cost = t.matmul(fea_b, fea_a.transpose(1,2))  #[B,T2,T1]
            norm = t.matmul(t.linalg.norm(fea_b, dim=-1, keep_dim=True),\
                            t.linalg.norm(fea_a, dim=-1, keep_dim=True).tranpose(1,2))  #[B,T2,T1]
            norm[norm<1e-8] = 1e-8
            cost = 1 - cost/norm  #[B,T2,T1]
        
        b_size = len_a.shape[0]
        loss = t.zeros(1)
        loss.requires_grad = True
        for i in range(b_size):
            size_a = len_a[i]  #valid length
            size_b = len_b[i]
            path_cost = float('inf') * t.ones(size_a+1)
            path_cost_prev = float('inf') * t.ones(size_a+1)  #[size_a+1]
            path_cost_prev[0] = 0.0
            
            cost_matrix = skew_matrix(cost[i])  #[T2+T1-1, T1]
            for j in range(size_a + size_b - 1):
                directions = t.cat([path_cost_prev[None, :-1], \
                                    path_cost[None, 1:] + self.warp_penalty, \
                                    path_cost[None, :-1] + self.warp_penalty],
                                   dim=0)  #[3,size_a]
                path_cost_next = cost_matrix[:size_a] + soft_minimum(directions, self.temp)  #[size_a]
                
                path_cost_next = t.cat([float('inf')*t.ones(1), path_cost_next])  #[size_a+1]
                path_cost, path_cost_prev = path_cost_next, path_cost
            
            loss += path_cost[-1]
        
        if self.reduction == 'mean':
            loss = loss.mean()
        
        return loss








