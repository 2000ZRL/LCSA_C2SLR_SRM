# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:56:39 2021

@author: Ronglai Zuo
Deformable ConvNets v1&v2 by mmcv
"""

import torch as t
import torch.nn as nn
from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
import torch.nn.functional as F


def gen_mask_from_pose(offset, heatmap):
    T, C, H, H = offset.shape
    assert C == 18
    
    # initial coords
    grid_x, grid_y = t.meshgrid(t.arange(H), t.arange(H))
    grid_x, grid_y = grid_x.repeat(9,1,1), grid_y.repeat(9,1,1)
    init_coords = t.cat([grid_x, grid_y], dim=0).float().cuda()  #[18,H,H]
    init_offset = t.tensor([-1,-1,-1,0,0,0,1,1,1,-1,0,1,-1,0,1,-1,0,1]).cuda().view(18,1,1)
    init_coords += init_offset
    
    # coords after shift
    shift_coords = offset.add(init_coords)  #[T,18,H,H]
    shift_cx, shift_cy = shift_coords.chunk(2, dim=1)  #[T,9,H,H]
    shift_coords = map(lambda x: t.stack(x, dim=0).permute(1,2,3,0), zip(shift_cx.permute(1,0,2,3), shift_cy.permute(1,0,2,3)))  #[T,H,H,2]*9
    
    # normalize coords into [-1,1]. See the docs of F.grid_sample. So we don't need to downsample heatmap
    shift_coords = list(map(lambda x: (x-t.tensor([(H-1)/2,(H-1)/2]).cuda()).div(t.tensor([(H-1)/2,(H-1)/2]).cuda()), shift_coords))
    
    # bilinear interpolate. Transpose the heatmap!!! (don't know why)
    mask = list(map(lambda x: F.grid_sample(heatmap.permute(0,1,3,2), x), shift_coords))
    mask = t.cat(mask, dim=1)  #[T,9,H,H]
    return mask
    # Mask can be generated by gaussian distribution between shifted coords 
    # and keypoints. Also, mask can be simply use the heatmap
    # should detach offsets or not?


class DCN_v1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 bias=False,
                 use_heatmap=None):
        super(DCN_v1, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels=in_channels,
                                     out_channels=2*kernel_size**2,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=True)
        self.core = DeformConv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  bias=bias)
        self.init_weights()
    
    def init_weights(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        
    def forward(self, x, heatmap=None):
        offset = self.conv_offset(x)
        return offset, None, self.core.forward(x, offset)


class DCN_v2(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1,
                 padding=1,
                 bias=False,
                 use_offset=True,
                 use_heatmap=None,
                 fmap_shape=None):
        super(DCN_v2, self).__init__()
        # if use_heatmap == 'replace':
        #     out_offset_channels = 2*kernel_size**2
        if use_offset and fmap_shape is None:
            out_offset_channels = 3*kernel_size**2
        else:
            out_offset_channels = kernel_size**2
        self.conv_offset = nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_offset_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=True)
        self.core = ModulatedDeformConv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=bias)
        self.init_weights()
        self.use_offset = use_offset
        self.use_heatmap = use_heatmap
        self.fmap_shape = fmap_shape
        
        if use_offset and fmap_shape is not None:
            self.num_basis = 15
            H, W = fmap_shape
            self.A = nn.Parameter(t.rand(self.num_basis, 2*kernel_size**2, 1).cuda())
            self.B = nn.Parameter(t.rand(self.num_basis, 1, H*W).cuda())
            # take feature map as inputs then output num_basis coefficients for each feature map
            self.coe_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                       nn.Flatten(start_dim=1),
                                       nn.Linear(in_channels, self.num_basis))
            self.coe_layer[-1].weight.data.zero_()
            self.coe_layer[-1].bias.data.zero_()
        
    def init_weights(self):
        # obey the paper Deformable ConvNets v2
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        
    def forward(self, x, heatmap=None):
        out = self.conv_offset(x)
        # if self.use_heatmap == 'replace':
        #     offset = out
        #     mask = gen_mask_from_pose(offset, heatmap).detach()  #detach?
        if not self.use_offset:
            mask = t.sigmoid(out)
            B, C, H, W = out.shape
            offset = t.zeros(B,2*C,H,W).cuda()
        elif self.fmap_shape is not None:
            mask = t.sigmoid(out)
            B, C, H, W = out.shape
            basis = self.A.matmul(self.B).view(self.num_basis, -1)  #[N_basis,18*HW]
            coe = self.coe_layer(x)  #[B,N_basis]
            offset = coe.matmul(basis).view(B,2*C,H,W)
        else:
            o1, o2, mask = t.chunk(out, 3, dim=1)
            offset = t.cat((o1,o2), dim=1)
            mask = t.sigmoid(mask)
            # mask = F.relu6(mask+3, inplace=True) / 6  #hard sigmoid
            if self.use_heatmap == 'prior':
                mask = mask.mul(gen_mask_from_pose(offset, heatmap).detach())
        return offset, mask, self.core.forward(x, offset, mask)


# Visual module by DCN
class DCN(nn.Module):
    def __init__(self, version='v1', num_dcn=5, use_heatmap=None):
        super(DCN, self).__init__()
        if version == 'v1':
            dcn_layer = DCN_v1
        elif version == 'v2':
            dcn_layer = DCN_v2
        else:
            raise ValueError('We only support v1 and v2')
        
        layers = []
        for i in range(9):
            if i < 9-num_dcn:
                layers.append(nn.Conv2d)
            else:
                layers.append(dcn_layer)
        
        self.DCN_stack = nn.ModuleList([
            layers[0](in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            layers[1](in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            layers[2](in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            layers[3](in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            layers[4](in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, use_heatmap=use_heatmap),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            layers[5](in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            layers[6](in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            layers[7](in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            layers[8](in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            ])
        
    def forward(self, x, heatmap=None):
        offset_lst = []
        mask_lst = []
        for layer in self.DCN_stack:
            if isinstance(layer, DCN_v1) or isinstance(layer, DCN_v2):
                offset, mask, x = layer(x, heatmap)
                offset_lst.append(offset)
                mask_lst.append(mask)
            else:
                x = layer(x)
        
        return offset_lst, mask_lst, x.flatten(1)
