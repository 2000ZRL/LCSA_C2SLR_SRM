"""
Feature disentangle for signer-independent setting
"""

import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
import math
import numpy as np

from modules.cbam import CBAM


def upd_signer_emb_bank(bank, signer_emb, signer):
    # signer_emb [T,C,H,W], signer [T]
    T = signer.shape[0]
    signer = signer.cpu().numpy()
    diff = np.diff(signer)
    split_size = []
    s = -1
    for s in np.where(diff!=0)[0]:
        if split_size == []:
            split_size.append(s+1)
        else:
            split_size.append(s+1-split_size[-1])
    split_size.append(T-s-1)

    split_emb = signer_emb.split(split_size, dim=0)
    idx = np.array(split_size).cumsum() - 1
    final_signer_emb = []
    for emb,i in zip(split_emb, idx):
        s = str(signer[i])
        prev_num = bank['num_'+s]
        num = emb.shape[0]

        upd_signer_emb = (emb.sum(dim=0) + bank[s]*prev_num) / (num+prev_num)
        final_signer_emb.append(upd_signer_emb.expand_as(emb))
        bank[s] = upd_signer_emb.detach()
        bank['num_'+s] += num
    return t.cat(final_signer_emb, dim=0)  #[T,C]


def stat_pool(frame_level_emb, len_video, sen_level=True):
    # len_video: list; frame_level_emb: [T,C,H,W]
    if not sen_level:
        mu = frame_level_emb.mean(dim=(-2,-1))  #[T,C]
        sigma = frame_level_emb.std(dim=(-2,-1))  #[T,C]
        return t.cat([mu, sigma], dim=-1)  #[T,2C]
    else:
        assert len_video.sum().item() == frame_level_emb.shape[0]
        len_video = list(len_video)
        frame_level_emb = frame_level_emb.mean(dim=(-2,-1))  #squeeze spatial dimension
        out = []
        for emb in frame_level_emb.split(len_video, dim=0):
            # emb [T,C]
            out.append(t.cat([emb.mean(dim=0), emb.std(dim=0)]))
        return t.stack(out, dim=0)  #[B,2C]


def expand_xvec(xvec, len_video, sen_level=True):
    if not sen_level:
        return xvec
    else:
        out = []
        i = 0
        for x in xvec.split(1, dim=0):
            out.append(x.expand(len_video[i], -1))
            i += 1
        return t.cat(out, dim=0)  #[T,C]


class GradReverse(Function):
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return -ctx.lambd * grad_output[0], None


class FeaDis(nn.Module):
    def __init__(self, num_channels, num_signers, fde_type):
        super(FeaDis, self).__init__()
        self.num_channels = num_channels
        self.num_signers = num_signers
        self.fde_type = fde_type
        self.num_iter = 0
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        if 'xvec' in self.fde_type:
            in_channels = num_channels*2
        else:
            in_channels = num_channels
        self.fc1 = nn.Sequential(nn.Linear(in_channels, num_channels),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.1))
        self.fc2 = nn.Sequential(nn.Linear(num_channels, num_channels),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.1))
        
        self.cls_weight = nn.Parameter(t.rand(num_signers, num_channels))
        self.cls_bias = nn.Parameter(t.zeros(num_signers))

        if 'dual_spat' in self.fde_type:
            # separate spatial and channel attention
            self.attention = CBAM(num_channels, 16, no_channel=True, channel_pool='max_softmax')
            self.attention2 = CBAM(num_channels, 16, no_channel=True, channel_pool='max_softmax')
        else:
            # self.attention = CBAM(num_channels, 16, parallel=True, channel_gate_type='conv')
            # self.attention = CBAM(num_channels, 16, parallel=True, channel_gate_type='mlp')
            # self.attention = CBAM(num_channels, 16, parallel=False, channel_gate_type='mlp')
            self.attention = CBAM(num_channels, 16, no_channel=True, channel_pool='max_softmax')
    

    def get_lambda(self):
        # get lambda for gradient reversal layer
        gamma = 10
        total_iter = 4376 / 2 * 45
        p = self.num_iter / total_iter
        return 2 / (1+math.exp(-gamma*p)) - 1
    

    def forward(self, x, signer=None, signer_emb_bank={}, len_video=None):
        # signer (T,)
        sg = sg2 = cg = out = None
        _, gates, _ = self.attention(x)
        cg, sg = gates  #channel [T,C,1,1] and spatial [T,1,H,W] gates
        if cg is None:
            out = x * sg
        elif sg is None:
            out = x * cg
        else:
            out = x * cg * sg

        if 'dual_spat' in self.fde_type:
            temp = out
            _, gates, _ = self.attention2(out)
            _, sg2 = gates
            out = out * sg2
            x = temp - out
        
        cam = channel_cam = signer_emb = None
        if self.training:
            self.num_iter += 1
            # x = out
            ori_iden = x
            if 'xvec' in self.fde_type:
                if 'rev' in self.fde_type:
                    x = GradReverse.apply(x, 1.0)
                x = stat_pool(x, len_video, sen_level=True)
                x = self.fc1(x)
                # signer_emb = expand_xvec(x, len_video, sen_level=True)
                x = self.fc2(x)  #[T,C] or [B,C]
                # x = self.mlp(x)
            else:
                if 'rev' in self.fde_type:
                    x = GradReverse.apply(x, 1.0)
                x = self.gap(x)
                x = x.flatten(1)  #[T,C]
                x = self.fc2(self.fc1(x))
                signer_emb = x
        
            x = F.linear(x, self.cls_weight, self.cls_bias)  #[T,S]
            
            if 'bank' in self.fde_type:
                signer_emb = upd_signer_emb_bank(signer_emb_bank, signer_emb, signer)

            if 'sim' in self.fde_type:
                q = signer_emb.unsqueeze(-1).unsqueeze(-1).expand_as(ori_iden)  #[T,C,H,W]
                cam = F.cosine_similarity(q, ori_iden).unsqueeze(1)   #[T,1,H,W]
                cam = (cam - cam.amin(dim=(-2,-1), keepdim=True)) / (cam.amax(dim=(-2,-1), keepdim=True) - cam.amin(dim=(-2,-1), keepdim=True) + 1e-8)
                # q = signer_emb.unsqueeze(1) / math.sqrt(C)  #[T,1,C]
                # cam = q.matmul(out.flatten(2))  #[T,1,HW]
                # recover spatial gates to logits (inverse sigmoid)
                # sg = t.log((sg+1e-8) / (1.0-sg+1e-8)).flatten(2)
            elif 'xvec' not in self.fde_type:
                cam_w = self.cls_weight.index_select(0, signer).unsqueeze(-1).unsqueeze(-1)  #[T,C,1,1]
                channel_cam = (cam_w - cam_w.amin(dim=1, keepdim=True)) / (cam_w.amax(dim=1, keepdim=True) - cam_w.amin(dim=1, keepdim=True) + 1e-8)
                cam = (ori_iden*cam_w).sum(dim=1, keepdim=True)  #[T,1,H,W]
                cam = (cam - cam.amin(dim=(-2,-1), keepdim=True)) / (cam.amax(dim=(-2,-1), keepdim=True) - cam.amin(dim=(-2,-1), keepdim=True) + 1e-8)

        return (sg,sg2,cam), (cg,channel_cam), signer_emb, x, out


class SignerClassifier(nn.Module):
    def __init__(self, num_channels, num_signers=8):
        super(SignerClassifier, self).__init__()
        self.classifier = nn.Linear(num_channels, num_signers)
    
    def forward(self, x):
        return self.classifier(x)


# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#     x = t.rand(2,100,28,28).cuda()
#     signer = t.cat([t.ones(1).long(), t.zeros(1).long()]).cuda()
#     net = FeaDis(100, 8).cuda()
#     s,c,x,o = net(x, signer)
#     print(x.shape, o.shape)