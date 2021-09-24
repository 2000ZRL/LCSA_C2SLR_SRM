# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:06:49 2020

@author: Ronglai ZUO
A stack of 2D convolution layers of "FCN for CSLR"
"""

import torch as t
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, pose=None, bef_which_pool=3, ratio=0.5):
        super(CNN, self).__init__()
        if pose == 'modality':
            input_channels = 3+1
        else:
            input_channels = 3
        self.CNN_stack = nn.ModuleList([
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            ])
        self.bef_which_pool = bef_which_pool
        self.ratio = ratio
        self.pose = pose
        
    def forward(self, x, heatmap=None):
        num_pool = 0
        if self.pose == 'modality':
            x = t.cat([x, heatmap], dim=1)
        for layer in self.CNN_stack:
            if isinstance(layer, nn.MaxPool2d):
                num_pool += 1
            if self.pose == 'filter' and num_pool == self.bef_which_pool:
                #pose filtering before the 3rd pooling layer
                n_channels = x.shape[1]
                if self.ratio < 1:
                    change, keep = x.split(int(n_channels*self.ratio), dim=1)
                    change = change.mul(heatmap)
                    x = t.cat([change, keep], dim=1)
                else:
                    x = x.mul(heatmap)
                num_pool += 1
            
            x = layer(x)
            
        x = t.flatten(x, 1)
        return None, None, x


class pose_stream(nn.Module):
    def __init__(self):
        super(pose_stream, self).__init__()
        # self.layers = nn.ModuleList([
        #     nn.Conv2d(in_channels=7, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),
        #     # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1,1))
        #     ])
        self.layers = nn.ModuleList([
            nn.Linear(14, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
            ])
        
    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        # x = t.flatten(x, 1)
        return x
