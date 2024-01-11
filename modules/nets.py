# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:06:21 2021

@author: Ronglai Zuo
Nets: VGGNet-11, ResNet-18, GoogLeNet, AlexNet
"""

import torch as t
import torch.nn as nn
from torch.nn import functional as F
from modules.cbam import CBAM
from modules.fde import FeaDis
from utils.utils import freeze_params
import math
# from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


#---------------------------------------------VGGNet----------------------------------------------
class VGG11(nn.Module):
    #VGG11 without FC layers
    def __init__(self, batch_norm=True, spatial_att=None, att_idx_lst=[], pool_type='avg', 
                cbam_no_channel=False, cbam_pool='max_avg', freeze=False, **kwargs):
        super(VGG11, self).__init__()
        self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.batch_norm = batch_norm
        self.spatial_att = spatial_att
        self.dcn_ver = 'v2'
        self.att_idx_lst = att_idx_lst
        self.freeze = freeze  # for self-supervised setting
        
        self.cbam_pool = cbam_pool
        self.cbam_no_channel = cbam_no_channel
        self.pool_type = pool_type
        if pool_type == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))  #7,7?
        else:
            if pool_type == 'gcnet-like':
                # GCNet-like attention pooling
                self.attpool = nn.Sequential(nn.Conv2d(512, 1, 1),
                                             nn.Softmax2d())
                # if use attention pooling, then remove the last max-pooling layer
            elif pool_type == 'avg_softmax':
                self.att_pool = nn.Softmax2d()
            elif pool_type == 'cbam-like':
                self.att_pool = nn.Sequential(nn.Conv2d(2,1,7,1,3,bias=False),
                                              nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
                                              nn.SoftMax2d())
            self.cfg = self.cfg[:-1]
        self.features = self.make_layers()

        self.fde_type = kwargs.pop('fde', None)
        if self.fde_type is not None:
            self.fde_mod = FeaDis(num_channels=512, num_signers=kwargs.pop('num_signers', 8), fde_type=self.fde_type)
    
    def make_layers(self):
        # if self.dcn_ver == 'v1':
        #     DCN = DCN_v1
        # elif self.dcn_ver == 'v2':
        #     DCN = DCN_v2
        DCN = None
        layers = []
        in_channels = 3
        num_conv, num_pool = 0, 0
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                num_pool += 1
            else:
                if self.spatial_att in ['dcn', 'dcn_nooff'] and num_conv in self.att_idx_lst:
                    conv2d = DCN(in_channels, v, kernel_size=3, padding=1, bias=False, use_offset=False if self.spatial_att=='dcn_nooff' else True, fmap_shape=[int(224/2**num_pool),int(224/2**num_pool)])
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                if self.spatial_att == 'cbam' and num_conv in self.att_idx_lst:
                    layers.append(CBAM(v, 16, no_channel=self.cbam_no_channel, channel_pool=self.cbam_pool))
                in_channels = v
                num_conv += 1
            if self.freeze and num_conv > max(self.att_idx_lst):
                # freeze all layer in layers
                for l in layers:
                    freeze_params(l)
        
        return nn.ModuleList(layers)
    
    def forward(self, x, **kwargs):
        heatmap = kwargs.pop('heatmap', None)
        signer = kwargs.pop('signer', None)
        signer_emb_bank = kwargs.pop('signer_emb_bank', {})
        len_video = kwargs.pop('len_video', None)
        offset_lst = []
        mask_lst = []
        num_conv_block = 0
        sg_sg2_cam = [None,None,None]; cg_ch_cam = [None,None]; signer_emb = signer_logits = None
        for layer in self.features:
            if isinstance(layer, CBAM):
                channel_weights, gates, x = layer(x)
                _, mask = gates
                mask_lst.append(mask)
                offset_lst.append(channel_weights)
            else:
                x = layer(x)
            
            if self.fde_type is not None and isinstance(layer, nn.ReLU):
                num_conv_block += 1
                if num_conv_block == self.att_idx_lst[0]+1:
                    sg_sg2_cam, cg_ch_cam, signer_emb, signer_logits, x = self.fde_mod(x, signer, signer_emb_bank, len_video)

            # heatmap filter
            # if heatmap is not None and isinstance(layer, nn.Conv2d):
            #     num_conv += 1
            #     if num_conv == 5:
            #         x = x.mul(heatmap)
        
        if self.pool_type == 'avg':
            x = self.avgpool(x)
        else:
            if self.pool_type == 'gcnet-like':
                w = self.attpool(x)
            elif self.pool_type == 'avg_softmax':
                w = self.attpool(x.mean(dim=1, keepdims=True))
            elif self.pool_type == 'cbam-like':
                w = self.attpool(t.cat((t.max(x,1)[0].unsqueeze(1), t.mean(x,1).unsqueeze(1)), dim=1))
            x = (w*x).sum(dim=(-2,-1))
        
        return {'offset_lst': offset_lst, 'mask_lst': mask_lst, 'output': x.flatten(1), 
                'cam': sg_sg2_cam, 'ch_cam': cg_ch_cam, 'signer_emb': signer_emb, 'signer_logits': signer_logits}


def vgg11(batch_norm=True, spatial_att=None, att_idx_lst=[], pool_type='avg', cbam_no_channel=False, cbam_pool='max_avg', freeze=False, pretrained=True, pre_model_path=None, **kwargs):
    model = VGG11(batch_norm, spatial_att, att_idx_lst, pool_type, cbam_no_channel, cbam_pool, freeze, fde=kwargs.pop('fde', None), num_signers=kwargs.pop('num_signers', 8))
    if pretrained:
        state_dict = t.load(pre_model_path)
        
        # modify keys in pretrained state_dict
        conv_layer = [0,4,8,11,15,18,22,25]
        if spatial_att in ['dcn', 'dcn_nooff']:
            for i in att_idx_lst[::-1]:
                idx = conv_layer[i]
                state_dict['features.'+str(idx)+'.core.weight'] = state_dict['features.'+str(idx)+'.weight']
                del state_dict['features.'+str(idx)+'.weight']
                # Since bias are illegal for deformable convnets, here we remove params of batchnorm layers followed deform layers
                # state_dict = dict(filter(lambda item: str(idx+1) not in item[0], state_dict.items()))
        
        elif spatial_att in ['ca', 'cbam']:
            offset = len(att_idx_lst)
            if att_idx_lst[-1] == len(conv_layer)-1:
                # add att block after the last conv layer
                offset -= 1
            for i in range(len(conv_layer)-1, att_idx_lst[0], -1):
                if i <= att_idx_lst[offset-1]:
                    offset -= 1
                idx = conv_layer[i]
                state_dict['features.'+str(idx+1+offset)+'.weight'] = state_dict['features.'+str(idx+1)+'.weight']
                del state_dict['features.'+str(idx+1)+'.weight']
                state_dict['features.'+str(idx+1+offset)+'.bias'] = state_dict['features.'+str(idx+1)+'.bias']
                del state_dict['features.'+str(idx+1)+'.bias']
                state_dict['features.'+str(idx+1+offset)+'.running_mean'] = state_dict['features.'+str(idx+1)+'.running_mean']
                del state_dict['features.'+str(idx+1)+'.running_mean']
                state_dict['features.'+str(idx+1+offset)+'.running_var'] = state_dict['features.'+str(idx+1)+'.running_var']
                del state_dict['features.'+str(idx+1)+'.running_var']
                state_dict['features.'+str(idx+offset)+'.weight'] = state_dict['features.'+str(idx)+'.weight']
                del state_dict['features.'+str(idx)+'.weight']
                state_dict['features.'+str(idx+offset)+'.bias'] = state_dict['features.'+str(idx)+'.bias']
                del state_dict['features.'+str(idx)+'.bias']
            assert offset == 1
        
        model.load_state_dict(state_dict, strict=False)
    return model


#---------------------------------------------------------ResNet---------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inchannels, outchannels, stride=1, downsample=None, norm_layer=None, spatial_att=None, cbam_no_channel=False, cbam_pool='max_avg'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(outchannels)
        self.relu = nn.ReLU(inplace=True)
        # if spatial_att == 'dcn':
        #     self.conv2 = DCN_v2(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
        # else:
        self.conv2 = nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(outchannels)
        
        if spatial_att == 'cbam':
            self.att_mod = CBAM(outchannels, 16, no_channel=cbam_no_channel, channel_pool=cbam_pool)
        self.downsample = downsample
        self.stride = stride
        self.spatial_att = spatial_att
    
    def forward(self, x):
        identity = x
        offset, mask = None, None
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.spatial_att == 'dcn':
            offset, mask, out = self.conv2(out)
        else:
            out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        if self.spatial_att in ['cbam', 'ca']:
            offset, mask, out = self.att_mod(out)
        
        return offset, mask, out


class ResNet_wo_fc(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2,2], norm_layer=None, zero_init_residual=True, spatial_att=None, cbam_no_channel=False, cbam_pool='max_avg'):
        super(ResNet_wo_fc, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inchannels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inchannels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], spatial_att=None)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, spatial_att=spatial_att, cbam_no_channel=cbam_no_channel, cbam_pool=cbam_pool)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, spatial_att=None)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, spatial_att=None)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # no fc layer
        
        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
    def _make_layer(self, block, channels, num_block, stride=1, spatial_att=None, cbam_no_channel=False, cbam_pool='max_avg'):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inchannels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inchannels, out_channels=channels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(channels*block.expansion),)
            
        layers = []
        layers.append(block(self.inchannels, channels, stride, downsample, norm_layer))
        self.inchannels = channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.inchannels, channels, norm_layer=norm_layer, spatial_att=spatial_att, cbam_no_channel=cbam_no_channel, cbam_pool=cbam_pool))
            
        return nn.ModuleList(layers)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        offset_lst, mask_lst = [], []
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for l in layer:
                offset, mask, x = l(x)
                offset_lst.append(offset)
                mask_lst.append(mask)
        
        x = self.avgpool(x)
        x = t.flatten(x, 1)  #[B, 512]?
        
        return {'offset_lst': offset_lst, 'mask_lst': mask_lst, 'output': x}
    
    def forward(self, x, **kwargs):
        return self._forward_impl(x)


def resnet18_wo_fc(block=BasicBlock, layers=[2,2,2,2], spatial_att=None, cbam_no_channel=False, cbam_pool='max_avg', pretrained=True, pre_model_path='', **kwargs):
    model = ResNet_wo_fc(block, layers, spatial_att=spatial_att, cbam_no_channel=cbam_no_channel, cbam_pool=cbam_pool)
    if pretrained:
        state_dict = t.load(pre_model_path)
        if spatial_att == 'dcn':
            for km in model.state_dict().keys():
                if 'core' in km:
                    k = ''.join(km.split('core.'))
                    state_dict[km] = state_dict[k]
                    del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
    
    return model


#---------------------------------------------9-layer CNN in FCN------------------------------------
class CNN(nn.Module):    
    def __init__(self, batch_norm=True, spatial_att=None, att_idx_lst=[], pool_type='avg', cbam_no_channel=False, cbam_pool='max_avg'):
        super(CNN, self).__init__()
        self.cfg = [32, 'M', 64, 'M', 64, 128, 'M', 128, 256, 'M', 256, 512, 'M', 512]
        self.batch_norm = batch_norm
        self.spatial_att = spatial_att
        self.att_idx_lst = att_idx_lst
        
        self.cbam_pool = cbam_pool
        self.cbam_no_channel = cbam_no_channel
        self.pool_type = pool_type
        if pool_type == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))  #7,7?
        else:
            if pool_type == 'gcnet-like':
                # GCNet-like attention pooling
                self.attpool = nn.Sequential(nn.Conv2d(512, 1, 1),
                                             nn.Softmax2d())
                # if use attention pooling, then remove the last max-pooling layer
            elif pool_type == 'avg_softmax':
                self.att_pool = nn.Softmax2d()
            elif pool_type == 'cbam-like':
                self.att_pool = nn.Sequential(nn.Conv2d(2,1,7,1,3,bias=False),
                                              nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
                                              nn.SoftMax2d())
            self.cfg  =self.cfg[:-1]
        self.features = self.make_layers()
    
    def make_layers(self):
        layers = []
        in_channels = 3
        num_conv, num_pool = 0, 0
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                num_pool += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                if self.spatial_att == 'cbam' and num_conv in self.att_idx_lst:
                    layers.append(CBAM(v, 16, no_channel=self.cbam_no_channel, channel_pool=self.cbam_pool))
                in_channels = v
                num_conv += 1
        
        return nn.ModuleList(layers)
    
    def forward(self, x, **kwargs):
        heatmap = kwargs.pop('heatmap', None)
        signer = kwargs.pop('signer', None)
        offset_lst = []
        mask_lst = []
        for layer in self.features:
            if isinstance(layer, CBAM):
                channel_weights, mask, x = layer(x)
                mask_lst.append(mask)
                offset_lst.append(channel_weights)
            else:
                x = layer(x)
        if self.pool_type == 'avg':
            x = self.avgpool(x)
        else:
            if self.pool_type == 'gcnet-like':
                w = self.attpool(x)
            elif self.pool_type == 'avg_softmax':
                w = self.attpool(x.mean(dim=1, keepdims=True))
            elif self.pool_type == 'cbam-like':
                w = self.attpool(t.cat((t.max(x,1)[0].unsqueeze(1), t.mean(x,1).unsqueeze(1)), dim=1))
            x = (w*x).sum(dim=(-2,-1))
        x = t.flatten(x, 1)
        return {'offset_lst': offset_lst, 'mask_lst': mask_lst, 'output': x}


def cnn9(batch_norm=True, spatial_att=None, att_idx_lst=[], pool_type='avg', cbam_no_channel=False, cbam_pool='max_avg', **kwargs):
    return CNN(batch_norm, spatial_att, att_idx_lst, pool_type, cbam_no_channel, cbam_pool)


#---------------------------------------------MobileNet_v2------------------------------------
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, spatial_att=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if spatial_att == 'cbam':
            att_mod = CBAM(hidden_dim, 16)

        if expand_ratio == 1:
            self.conv = nn.ModuleList([
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
        else:
            self.conv = nn.ModuleList([
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # coordinate attention
                att_mod if spatial_att is not None else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

    def forward(self, x):
        iden = x
        mask = None
        for layer in self.conv:
            if isinstance(layer, CBAM):
                mask, x = layer(x)
            else:
                x = layer(x)
        if self.identity:
            return mask, x + iden
        else:
            return mask, x

class MBV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., emb_size=1280, spatial_att=None):
        super(MBV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # r, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for r, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, r, spatial_att=spatial_att))
                input_channel = output_channel
        self.features = nn.ModuleList(layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Sequential(
        #         nn.Dropout(0.1),
        #         nn.Linear(output_channel, num_classes)
        #         )
        
        if emb_size != 1280:
            self.fc = nn.Linear(1280, emb_size)
        self.emb_size = emb_size

        self._initialize_weights()

    def forward(self, x, **kwargs):
        heatmap = kwargs.pop('heatmap', None)
        signer = kwargs.pop('signer', None)
        mask_lst = []
        for layer in self.features:
            if isinstance(layer, InvertedResidual):
                mask, x = layer(x)
                mask_lst.append(mask)
            else:
                x = layer(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.emb_size != 1280:
            x = self.fc(x)

        return [], mask_lst, x

    def _initialize_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                #print(m.weight.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mb_v2(emb_size=1280, spatial_att=None, pretrained=True, pre_model_path=None):
    model = MBV2(emb_size=emb_size, spatial_att=spatial_att)
    if pretrained:
        state_dict = t.load(pre_model_path)
        model.load_state_dict(state_dict, strict=False)
    return model


#---------------------------------------------MobileNet_v3--------------------------------------
# class MobileNet_v3(nn.Module):
#     def __init__(self, variant='large', spatial_att=None):
#         super(MobileNet_v3, self).__init__()
#         if variant == 'large':
#             self.model = mobilenet_v3_large(pretrained=True)
#             self.block_idx = '6'
#             if spatial_att == 'cbam':
#                 self.att_mod = CBAM(40, 2, no_channel=True, channel_pool='max_softmax')
#         elif variant == 'small':
#             self.model = mobilenet_v3_small(pretrained=True)
#             self.block_idx = '3'
#             if spatial_att == 'cbam':
#                 self.att_mod = CBAM(24, 2, no_channel=True, channel_pool='max_softmax')
#         else:
#             raise ValueError
#         self.spatial_att = spatial_att
#         self.model.classifier = nn.Identity()

    
#     def forward(self, x, **kwargs):
#         offset_lst = []
#         mask_lst = []
#         sg_sg2_cam = [None,None,None]; cg_ch_cam = [None,None]; signer_emb = signer_logits = None
#         for name, layer in self.model.features.named_children():
#             x = layer(x)
#             if self.spatial_att == 'cbam' and name == self.block_idx:
#                 #after the 6th block
#                 channel_weights, gates, x = self.att_mod(x)
#                 _, mask = gates
#                 mask_lst.append(mask)
#                 offset_lst.append(channel_weights)
#         x = self.model.avgpool(x)

#         return {'offset_lst': offset_lst, 'mask_lst': mask_lst, 'output': x.flatten(1), 
#                 'cam': sg_sg2_cam, 'ch_cam': cg_ch_cam, 'signer_emb': signer_emb, 'signer_logits': signer_logits}
    
#---------------------------------------------GoogLeNet---------------------------------------
class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes = 1000,
        aux_logits = False,
        transform_input = False,
        init_weights = None,
        blocks = None,
        spatial_att=None
    ):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            init_weights = True
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(1024, num_classes)
        # self.down = nn.Linear(1024, 512)

        if init_weights:
            self._initialize_weights()
        self.spatial_att = spatial_att
        if spatial_att == 'cbam':
            self.att_mod = CBAM(480, 16, no_channel=True, channel_pool='max_softmax')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = t.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with t.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = t.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = t.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = t.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = t.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        offset_lst = []
        mask_lst = []
        sg_sg2_cam = [None,None,None]; cg_ch_cam = [None,None]; signer_emb = signer_logits = None

        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        if self.spatial_att == 'cbam':
            channel_weights, gates, x = self.att_mod(x)
            _, mask = gates
            mask_lst.append(mask)
            offset_lst.append(channel_weights)
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1 = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2 = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = t.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        # x = self.fc(x)
        # x = self.down(x)
        # N x 512
        return {'offset_lst': offset_lst, 'mask_lst': mask_lst, 'output': x, 
                'cam': sg_sg2_cam, 'ch_cam': cg_ch_cam, 'signer_emb': signer_emb, 'signer_logits': signer_logits}

    def forward(self, x, **kwargs):
        x = self._transform_input(x)
        op_dict = self._forward(x)
        return op_dict


class Inception(nn.Module):

    def __init__(
        self,
        in_channels,
        ch1x1,
        ch3x3red,
        ch3x3,
        ch5x5red,
        ch5x5,
        pool_proj,
        conv_block = None
    ):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return t.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        conv_block = None
    ):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = t.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        **kwargs
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def googlenet(pretrained=True, pre_model_path=None, spatial_att=None):
    model = GoogLeNet(spatial_att=spatial_att)
    if pretrained:
        state_dict = t.load(pre_model_path)
        model.load_state_dict(state_dict, strict=False)
    return model

