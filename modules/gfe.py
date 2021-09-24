# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:19:45 2020

@author: Ronglai ZUO
Given label and prediction map (gloss+blank probability), find the best alignment proposal
"""

import numpy as np
import torch as t
import torch.nn as nn


def get_best_alignment(prob, len_video, label, len_label, video_id, blank_id, proposal_dict):
    '''
    Parameters
    ----------
    prob : cuda Tensor 
        with shape [B,T,C=1233]
    len_video : cuda Tensor
        with shape [B]
    label : cuda Tensor
        with shape [sum of len_label]
    len_label : cuda Tensor
        with shape [B]
    blank_id : int
        default=1232

    Returns
        pi : numpy ndarray with shape [sum of len_video]
    -------
    Alignment proposal with highest probability
    Reference: A Deep Neural Framework for Continuous Sign Language Recognition by Iterative Training
    '''
    assert blank_id == prob.shape[-1]-1
    prob = prob.cpu().detach().numpy()
    len_video = len_video.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    len_label = len_label.cpu().detach().numpy()
    batch_size = prob.shape[0]
    
    start = 0
    start_frame = 0
    pi = t.zeros(int(np.sum(len_video)), dtype=t.long)  #[sum of len_video]
    balance_ratio = np.zeros(batch_size)
    for i in range(batch_size):
        sample_label = label[start: start+len_label[i]]
        K = len_video[i]
        U = len_label[i]
        ex_U = 2*U+1  #lenght after inserting blank
        ex_label = blank_id * np.ones([ex_U+1], dtype=np.int32)
        for j in range(U):
            ex_label[2*j+2] = sample_label[j]   #remain space for u=0 and start from u=1
        
        #initialize alpha
        alpha = np.zeros((K, ex_U+1))
        alpha[0][1] = prob[i][0][ex_label[1]]
        alpha[0][2] = prob[i][0][ex_label[2]]
        
        #calculate alpha recursively
        g = np.zeros(ex_U+1, dtype=np.int32)
        for k in range(1, K):
            for u in range(1, ex_U+1):
                if ex_label[u]==blank_id or ex_label[u] == ex_label[u-2]:
                    g[u] = u-1
                else:
                    g[u] = u-2
                    
                assert g[u] >= 0
                alpha[k][u] = prob[i][k][ex_label[u]] * max(alpha[k-1][g[u]:u+1])
                
        #backtrack
        gamma = ex_U-1+np.argmax(alpha[K-1][ex_U-1:ex_U+1])
        pi[start_frame+K-1] = ex_label[gamma]
        if pi[start_frame+K-1] != blank_id:
            balance_ratio[i] += 1
            
        for k in range(K-2, -1, -1):
            gamma = g[gamma]+np.argmax(alpha[k][g[gamma]:gamma+1])
            pi[start_frame+k] = ex_label[gamma]
            if pi[start_frame+K-1] != blank_id:
                balance_ratio[i] += 1
        
        balance_ratio[i] /= K
        sample_id = ''.join(video_id[i])
        proposal_dict[sample_id] = pi[start_frame:start_frame+K]
        proposal_dict[sample_id+'_br'] = balance_ratio[i]
        
        start += U
        start_frame += K
        
    assert start == int(np.sum(len_label))
    assert start_frame == int(np.sum(len_video))


class GFE(nn.Module):
    def __init__(self, 
                 inchannels=512,
                 voc_size=1233
                 ):
        super(GFE, self).__init__()
        self.fc = nn.Linear(inchannels, voc_size)
    
    def forward(self, x):
        '''
        Parameters
        ----------
        x : cuda Tensor
            with shape [B, max_T, C]
            
        Returns
        -------
        log_prob : cuda Tensor
            with shape [B, max_T, voc_size]
        '''
        x = self.fc(x)
        return x.log_softmax(-1)
    

# if __name__ == '__main__':
#     voc_size = 20
#     gfe = GFE(inchannels=10, voc_size=voc_size)
#     t.manual_seed(88)
#     x = t.rand(4,7,10)
#     log_prob = gfe(x)
#     log_prob = log_prob.transpose(1,2)
#     target = t.ones(4,7,dtype=t.long)   #split and pad_sequence
#     target[0, 3:] = -1
#     target[2, 5:] = -1
#     target[1, 2:] = voc_size-1
#     target[3, 4:] = voc_size-1
#     weight = t.ones(voc_size)
#     weight[voc_size-1] = 0.5
#     criterion = nn.NLLLoss(weight=weight, ignore_index=-1)
#     loss = criterion(log_prob, target)
#     print(loss)
    

































