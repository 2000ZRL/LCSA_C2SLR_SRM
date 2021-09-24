# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:57:26 2020

@author: Ronglai ZUO
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from tfmer.embeddings import MaskedMean


class SE_block(nn.Module):
    """
    SE block. See ContextNet
    x: [B, M, D=512]
    """
    
    def __init__(self, input_size=512, win_size=16, time_ratio=8, fea_ratio=1, stride=16,
                 time=True, act_fun='softsign'):
        super(SE_block, self).__init__()
        self.size = input_size
        self.t_ratio = time_ratio
        self.f_ratio = fea_ratio
        self.win_size = win_size
        if stride == 1:
            self.stride = stride
            assert self.win_size % 2 == 1
        else:
            self.stride = self.win_size
        self.time = time  #True means do SE along time axis, False means do SE along feature axis
        self.act_fun = act_fun
        
        #layers
        if self.time:
            self.fc1 = nn.Linear(self.size, self.size//self.t_ratio)
            self.fc2 = nn.Linear(self.size//self.t_ratio, self.size)
        else: 
            self.fc1 = nn.Linear(self.win_size, self.win_size//self.f_ratio)
            self.fc2 = nn.Linear(self.win_size//self.f_ratio, self.win_size)
        if self.act_fun == 'softsign':
            self.act1 = nn.Softsign()
            self.act2 = nn.Softsign()
        self.sigmoid = nn.Sigmoid()
        self.layers = nn.ModuleList([self.fc1, self.act1, self.fc2, self.act2, self.sigmoid])
        
    def forward(self, x):
        time_d = x.shape[1]
        
        #padding to make time_d can be exactly divided by window_size
        if self.stride == self.win_size:
            if time_d % self.win_size != 0:
                time_d_pad = (time_d//self.win_size + 1) * self.win_size
                pad_shape = ((time_d_pad-time_d)//2, time_d_pad-time_d-(time_d_pad-time_d)//2)
                x = F.pad(x.transpose(1,2), pad=pad_shape, mode='replicate').transpose(1,2)
                assert x.shape[1] % self.win_size == 0
                
            else:
                time_d_pad = time_d
                
        else:
            time_d_pad = time_d + self.win_size - 1
            pad_shape = (self.win_size//2, self.win_size//2)
            x = F.pad(x.transpose(1,2), pad=pad_shape, mode='replicate').transpose(1,2)
            
        weights = torch.zeros(x.shape).cuda()
        
        if self.time == True:
            i = 0 
            while (i+self.win_size <= time_d_pad):
                one_window_weight = torch.mean(x[:, i:i+self.win_size, :], dim=1, keepdim=True)
                for layer in self.layers:
                    one_window_weight = layer(one_window_weight)
                
                if self.stride == self.win_size:
                    weights[:, i:i+self.stride, :] = one_window_weight
                else:
                    weights[:, i+self.win_size//2:i+self.win_size//2+self.stride, :] = one_window_weight
                
                i += self.stride
            
            x = torch.mul(x, weights)
            # if i < time_d:
            #     weights = torch.mean(x[:, i:time_d, :], dim=1, keepdim=True)
            #     weights = self.layers(weights)
            #     x[:, i:i+self.win_size, :] = torch.mul(x[:, i:time_d, :], weights)
            
        else:  #feature axis
            weights = torch.mean(x, dim=-1, keepdim=True).transpose(1,2)  #[B,1,M]
            for layer in self.layers:
                weights = layer(weights)
            x = torch.mul(x, weights)
            
        return x[:, (time_d_pad-time_d)//2:(time_d_pad-time_d)//2+time_d, :]
    
    
class SE_block_time(nn.Module):
    """
    SE block. See ContextNet
    x: [B, M, D=512]
    """
    
    def __init__(self, input_size=512, ratio=8, act_fun='softsign'):
        super(SE_block_time, self).__init__()
        self.size = input_size
        self.ratio = ratio
        self.act_fun = act_fun
        
        #layers
        #self.global_avg_pooling = nn.AvgPool1d(kernel_size=self.kernel_size)
        self.fc1 = nn.Linear(self.size, self.size//self.ratio)
        self.fc2 = nn.Linear(self.size//self.ratio, self.size)
        if self.act_fun == 'softsign':
            self.act1 = nn.Softsign()
            self.act2 = nn.Softsign()
        elif self.act_fun == 'relu':
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x_ip, mask, need_mid=False):
        #x_ip [B,max_T,C]
        x = MaskedMean(x_ip.transpose(1,2), mask)
        x = x.unsqueeze(1)  #[B,1,C]
        
        x = self.fc1(x)  #[B,1,C//r]
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        
        if need_mid:
            return x
        
        x = self.sigmoid(x)  #[B,1,C]
        x = torch.mul(x, x_ip)  #[B,max_T,C]
        return x
    
    
class SE_block_conv(nn.Module):
    #use 1d-conv rather than fc
    def __init__(self, kernel_size=5, mode='temporal'):
        #temporal: True for temporal attention, False for channel attention
        super(SE_block_conv, self).__init__()
        self.mode = mode
        
        if mode == 'mix':
            #we need two conv layer
            self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
            self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        elif mode == 'mix_2d':
            self.conv2d = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        else:
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x_ip, mask):
        #x_ip [B,max_T,C]
        #mask [B,1,max_T]  True for paddings.
        if self.mode == 'channel':
            #channel attention, masked mean and vanilla conv
            x = MaskedMean(x_ip.transpose(1,2), mask)
            x = x.unsqueeze(1)  #[B,1,C]
            
            x = self.conv(x)  #[B,1,C]
            x = self.sigmoid(x)
            x = x.mul(x_ip)  #[B,max_T,C]
            
        elif self.mode == 'temporal':
            #temporal attention, vanilla mean and masked conv
            x = torch.mean(x_ip, dim=-1)  #[B,max_T]
            x = x.unsqueeze(1)  #[B,1,max_T]
            
            x = self.conv(x)  #[B,1,MAX_T]
            x = x.mul(mask<=0)  #[B,1,MAX_T]
            
            x = self.sigmoid(x)
            x = x.transpose(1,2).mul(x_ip)  #[B,MAX_T,C]
        
        elif self.mode == 'mix':
            #channel
            x_ch = MaskedMean(x_ip.transpose(1,2), mask)
            x_ch = x_ch.unsqueeze(1)  #[B,1,C]
            x_ch = self.conv1(x_ch)  #[B,1,C]
            
            #temporal
            x = torch.mean(x_ip, dim=-1)  #[B,max_T]
            x = x.unsqueeze(1)  #[B,1,max_T]
            x = self.conv2(x)  #[B,1,MAX_T]
            x = x.mul(mask<=0)  #[B,1,MAX_T]
            
            #mix
            # x = x.transpose(1,2).matmul(x_ch)  #[B,MAX_T,C]
            x = torch.add(x.transpose(1,2), x_ch)  #[B,MAX_T,C]
            x = self.sigmoid(x)
            x = x.mul(x_ip)
        
        elif self.mode == 'mix_2d':
            #channel
            x_ch = MaskedMean(x_ip.transpose(1,2), mask)
            x_ch = x_ch.unsqueeze(1)  #[B,1,C]
            
            #temporal
            x = torch.mean(x_ip, dim=-1)  #[B,max_T]
            x = x.unsqueeze(1)  #[B,1,max_T]
            x = x.mul(mask<=0)  #[B,1,MAX_T]
            
            #mix
            x = torch.add(x.transpose(1,2), x_ch).unsqueeze(1)  #[B,1,MAX_T,C]
            x = self.conv2d(x).squeeze(1)  #[B,MAX_T,C]
            x = self.sigmoid(x)
            x = x.mul(x_ip)
        
        return x
