# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:17:53 2021

@author: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], gate_type='mlp'):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.gate_type = gate_type
        if gate_type == 'mlp':
            self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        elif gate_type == 'conv':
            #ECA-Net
            self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)

        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                if self.gate_type == 'mlp':
                    channel_att_raw = self.mlp( avg_pool )
                elif self.gate_type == 'conv':
                    channel_att_raw = self.conv(avg_pool.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).squeeze()

            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                if self.gate_type == 'mlp':
                    channel_att_raw = self.mlp( max_pool )
                elif self.gate_type == 'conv':
                    channel_att_raw = self.conv(max_pool.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).squeeze()
            
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3)#.expand_as(x)
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x, pool_type='max_avg'):
        if pool_type == 'max_avg':
            return None, torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        else:
            w = x.mean(dim=(-2,-1), keepdims=True).softmax(dim=1)  # return w for visualization
            w_avg = (w*x).sum(dim=1, keepdims=True)
            if pool_type == 'softmax':
                return w, w_avg
            elif pool_type == 'max_softmax':
                return w, torch.cat((torch.max(x,1)[0].unsqueeze(1), w_avg), dim=1)

class SpatialGate(nn.Module):
    def __init__(self, channel_pool='max_avg'):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.channel_pool = channel_pool
        self.compress = ChannelPool()
        self.spatial = BasicConv(1 if channel_pool=='softmax' else 2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        channel_weights, x_compress = self.compress(x, self.channel_pool)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return channel_weights, scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], 
                no_channel=False, channel_pool='max_avg', parallel=False, channel_gate_type='mlp', no_spatial=False):
        super(CBAM, self).__init__()
        self.no_channel = no_channel
        self.no_spatial = no_spatial
        self.parallel = parallel
        if not no_channel:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, channel_gate_type)
        if not no_spatial: 
            self.SpatialGate = SpatialGate(channel_pool)

    def forward(self, x):
        if not self.parallel:
            if not self.no_channel:
                cg = self.ChannelGate(x)
                out = x * cg
                channel_weights, sg = self.SpatialGate(out)
                out = out * sg
            else:
                cg = None
                channel_weights, sg = self.SpatialGate(x)
                out = x * sg
        
        else:
            out = cg = channel_weights = sg = None
            if not self.no_channel:
                cg = self.ChannelGate(x)  #[T,C,H,W]
            if not self.no_spatial:
                channel_weights, sg = self.SpatialGate(x)  #[T,1,H,W]
        
        return channel_weights, [cg, sg], out


# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#     x = torch.rand(2,512,28,28).cuda()
#     net = CBAM(512, 16, parallel=True, channel_gate_type='conv')
#     net = net.cuda()
#     _, gates, _ = net(x)
#     cg, sg = gates
#     print(cg.shape, sg.shape)