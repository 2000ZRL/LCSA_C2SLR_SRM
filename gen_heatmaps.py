# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:59:21 2021

@author: 14048
Test
"""

# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import argparse
import os
import logging
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from torchvision import transforms

from hrnet.default_cfg import _C as cfg
from hrnet.default_cfg import update_config
from hrnet.pose_hrnet import get_pose_net
from tqdm import tqdm
import cv2
import numpy as np

from phoenix_datasets import PhoenixVideoTextDataset, PhoenixTVideoTextDataset, CSLVideoTextDataset, TVBVideoTextDataset, CSLDailyVideoTextDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='hrnet/w32_256x256_adam_lr1e-3.yaml',
                        type=str)
    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    
    parser.add_argument('--dataset', type=str, default='2014', choices=['2014', '2014T', 'csl1', 'csl2', 'tvb', 'csl-daily'])
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--img_per_iter', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=2)

    args = parser.parse_args()
    return args


def create_dataloader(dset_name='2014', split='train', bsize=1):
    dset_dict = {'2014': {'cls': PhoenixVideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-multisigner', 'mean': [0.5372,0.5273,0.5195], 'hmap_mean': [0.0236, 0.0250, 0.0164, 0.0283, 0.0305, 0.0240, 0.0564]},
                 '2014T': {'cls': PhoenixTVideoTextDataset, 'root': '../../data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T', 'mean': [0.5372,0.5273,0.5195], 'hmap_mean': []},
                 'csl1': {'cls': CSLVideoTextDataset, 'root': ('../../data/ustc-csl', 'split_1.txt'), 'mean': [0.5827, 0.5742, 0.5768], 'hmap_mean': []},
                 'csl2': {'cls': CSLVideoTextDataset, 'root': ('../../data/ustc-csl', 'split_2.txt'), 'mean': [0.5827, 0.5742, 0.5768], 'hmap_mean': []},
                 'tvb': {'cls': TVBVideoTextDataset, 'root': '../../data/tvb', 'mean': [0.4706, 0.5277, 0.5247], 'hmap_mean': []},
                 'csl-daily': {'cls': CSLDailyVideoTextDataset, 'root': '../../data/csl-daily', 'mean': [0.6868, 0.6655, 0.6375]}}
    
    args_data = {'dataset': dset_name, 'aug_type': 'random_drop', 'max_len': 999, 'p_drop': 0, 'resize_shape': [256,256], 'crop_shape': [256,256]}

    dset_dict = dset_dict[dset_name]
    dset = dset_dict['cls'](args=args_data,
                            root=dset_dict['root'],
                            split=split,
                            normalized_mean=dset_dict['mean'],
                            use_random=False,
                            temp_scale_ratio=0)
    
    dataloader = DataLoader(dset, 
                            bsize,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=dset.collate_fn,
                            drop_last=False)
    
    return dset_dict['root'], dataloader
    

def main():
    args = parse_args()
    update_config(cfg, args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = get_pose_net(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        print('load model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)

    model = model.cuda()
    model = model.eval()

    # Data loading code
    root, valid_loader = create_dataloader(dset_name=args.dataset, split=args.split, bsize=1)
    
    # evaluate on validation set
    # if 'csl' in args.dataset and 'csl-daily' not in args.dataset:
    #     path = os.path.join(root[0], 'heatmaps_7', args.split)
    # else:
    #     path = os.path.join(root, 'heatmaps_7')
    
    if args.dataset == '2014T':
        path = os.path.join('/3tdisk/shared/rzuo/PHOENIX-2014-T', 'heatmaps_hrnet_mpii_9', args.split)
    elif args.dataset == 'csl-daily':
        path = os.path.join('/3tdisk/shared/rzuo/CSL-Daily', 'heatmaps_hrnet_mpii_9')

    if not os.path.exists(path):
        os.makedirs(path)
    
    with torch.no_grad():
        channel_mean = np.zeros(7, dtype=np.float64)
        total_len = 0
        for i, batch_data in tqdm(enumerate(valid_loader), desc='[Generating heatmaps of {:s} of {:s}]'.format(args.split, args.dataset)):
            video = torch.cat(batch_data['video']).cuda()  #[sumT,3,256,256]
            len_video = batch_data['len_video'][0]
            video_id = batch_data['id'][0]
            fname = os.path.join(path, ''.join(video_id))
            
            #inference
            #7 - thorax, 8 - upper neck
            #9 - head top, 
            #10 - r wrist, 11 - r elbow, 12 - r shoulder, 
            #13 - l shoulder, 14 - l elbow, 15 - l wrist
            heatmaps = model(video)
            heatmaps = heatmaps.detach().cpu().numpy()[:, 7:, ...]
            # coords = argmax(heatmaps)
            # heatmaps = lin_normalize(heatmaps)
            # print(coords[0,6,:], coords[0,0,:])
            
            # data = np.load(fname+'.npz')
            # heatmaps, finer_coords = data['heatmaps'], data['finer_coords']
            # finer_coords = argmax(heatmaps)
            # print(heatmaps.shape, finer_coords.shape, len_video)
            assert heatmaps.shape == (len_video,9,64,64) #and finer_coords.shape == (len_video,7,2)
            # channel_mean += heatmaps.sum(axis=(0,2,3))/(64*64)
            # total_len += len_video
            # vis(heatmaps)
            
            #save
            # if args.dataset in ['csl', 'CFSW']:
            #     dirs = os.path.join('/', *fname.split('/')[:-1])
            #     if not os.path.exists(dirs):
            #         os.makedirs(dirs)
            np.savez_compressed(fname+'.npz', heatmaps=heatmaps)
            
            # visualize
            # coords = finer_coords[:, (0,6,1), :]
            # heatmaps = heatmaps[:, (0,6,1), ...]
            
            # if i==0:
            #     hmaps = gen_gaussian_hmap(torch.from_numpy(coords), hmap_shape=[224,224], hmap_num=3, gamma=3)
            #     vis(video.detach().cpu(), hmaps)
            #     break
            # else:
            #     continue
        
        # channel_mean /= total_len
        # print(channel_mean)
        # np.savez(os.path.join(path,'channel_mean.npz'), mean=channel_mean)
        

def lin_normalize(hmaps):
    min = hmaps.min(axis=(-2,-1), keepdims=True)
    max = hmaps.max(axis=(-2,-1), keepdims=True)
    hmaps = (hmaps - min) / (max - min + 1e-6)
    return hmaps

def argmax(heatmap):
    T, C, H, W = heatmap.shape
    
    # simple argmax
    # coords = heatmap.reshape([T,C,-1])
    # coords = coords.argmax(axis=-1)
    # coords = np.stack([coords, np.zeros([T,C])], axis=2)
    # coords[..., 1] = (coords[..., 0] % W) / (W-1)
    # coords[..., 0] = (coords[..., 0] // W) / (H-1)
    
    # finer 
    coords_lst = []
    for i in range(C):
        if i==1:
            # right wrist
            mask = np.zeros([H,W]).astype(bool)
            mask[H//4:, :W//2] = True
            real_w = W//2
        elif i==6:
            # left wrist
            mask = np.zeros([H,W]).astype(bool)
            mask[H//4:, W//2:] = True
            real_w = W-W//2
        else:
            mask = np.ones([H,W]).astype(bool)
            real_w = W
        
        hmap = heatmap[:, i, ...]  #[T,H,W]
        hmap = hmap.transpose(1,2,0)
        sel_hmap = hmap[mask]  #[N,T]
        coords = sel_hmap.argmax(axis=0) #[T]
        coords = np.stack([coords, np.zeros(T)], axis=1)  #[T,2]
        
        if i==1:
            coords[:, 1] = (coords[:, 0] % real_w) / (W-1)
            coords[:, 0] = (coords[:, 0] // real_w + H//4) / (H-1)
        elif i==6:
            coords[:, 1] = (coords[:, 0] % real_w + W//2) / (W-1)
            coords[:, 0] = (coords[:, 0] // real_w + H//4) / (H-1)
        else:
            coords[:, 1] = (coords[:, 0] % real_w) / (W-1)
            coords[:, 0] = (coords[:, 0] // real_w) / (H-1)
            
        coords_lst.append(coords)
    coords = np.stack(coords_lst, axis=1)  #[T,C,2]
    
    return coords

def soft_argmax(heatmap):
    # Pls refer to STMC
    H, W = heatmap.shape[-2:]
    hmap = np.exp(heatmap - heatmap.max(axis=(-2,-1), keepdims=True))
    hmap /= hmap.sum(axis=(-2,-1), keepdims=True)
    x_prob = hmap.sum(axis=-1)  #[T,3,H]
    y_prob = hmap.sum(axis=-2)  #[T,3,W]
    print(x_prob[0][0][:])
    x_range = np.arange(H).astype(np.float32)
    y_range = np.arange(W).astype(np.float32)
    x = (x_prob*x_range).sum(axis=-1)  #[T,3]
    y = (y_prob*y_range).sum(axis=-1)  #[T,3]
    return np.stack([x/(H-1), y/(W-1)], axis=2)

def gen_gaussian_hmap(coords, hmap_shape, hmap_num=3, gamma=14):
    H, W = hmap_shape
    sigma = H/gamma
    T = coords.shape[0]
    x, y = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack([x,y], dim=2)  #[H,H,2]
    grid = grid.repeat((T,1,1,1)).permute(1,2,0,3)  #[H,H,T,2]
    hmap = [torch.exp(-((grid-(c.squeeze()*(H-1)))**2).sum(dim=-1) / (2*sigma**2)) for c in coords.chunk(hmap_num, dim=1)]  #[H,H,T]
    hmap = torch.stack(hmap, dim=0).permute(3,0,1,2)  #[T,3,H,H]
    return hmap

def vis(video, hmaps, normalize=None):
    hmaps = np.array(hmaps)
    assert hmaps.shape[1] == 3
    
    if normalize is not None:
        head = normalize(hmaps[:, 0, ...].copy())
        l_wrist = normalize(hmaps[:, 1, ...].copy())
        r_wrist = normalize(hmaps[:, 2, ...].copy())
        hmaps = normalize(hmaps)
    else:
        head = hmaps[:, 0, ...].copy()
        l_wrist = hmaps[:, 1, ...].copy()
        r_wrist = hmaps[:, 2, ...].copy()
    
    head = np.uint8(255*head)
    l_wrist = np.uint8(255*l_wrist)
    r_wrist = np.uint8(255*r_wrist)
    
    mean = torch.tensor([0.5372,0.5273,0.5195]).reshape(1,3,1,1)
    video += mean
    # video = transforms.functional.resize(video, hmaps.shape[-2:])
    video = np.uint8(255*video.numpy()).transpose(2,3,1,0)
    hmaps = hmaps.max(axis=1)
    hmaps = np.uint8(255*hmaps)
    for i in range(0, hmaps.shape[0], 2):
        # original heatmap
        # plt.figure(0)
        # plt.imshow(cv2.applyColorMap(head[i, ...], cv2.COLORMAP_JET)[..., ::-1])
        # plt.axis('off')
        # plt.savefig('vis_res/img_orihmap_gauhmap/'+str(i)+'_head.jpg', bbox_inches='tight', pad_inches=0)
        
        # plt.figure(1)
        # plt.imshow(cv2.applyColorMap(l_wrist[i, ...], cv2.COLORMAP_JET)[..., ::-1])
        # plt.axis('off')
        # plt.savefig('vis_res/img_orihmap_gauhmap/'+str(i)+'_lwrist.jpg', bbox_inches='tight', pad_inches=0)
        
        # plt.figure(2)
        # plt.imshow(cv2.applyColorMap(r_wrist[i, ...], cv2.COLORMAP_JET)[..., ::-1])
        # plt.axis('off')
        # plt.savefig('vis_res/img_orihmap_gauhmap/'+str(i)+'_rwrist.jpg', bbox_inches='tight', pad_inches=0)

        # # raw image
        # plt.figure(3)
        # plt.imshow(video[..., i])
        # plt.axis('off')
        # plt.savefig('vis_res/img_orihmap_gauhmap/'+str(i)+'_img.jpg', bbox_inches='tight', pad_inches=0)

        # gaussian heatmap
        # plt.figure(1)
        # plt.imshow(cv2.applyColorMap(hmaps[i, ...], cv2.COLORMAP_JET)[..., ::-1])
        # plt.axis('off')
        # plt.savefig('vis_res/img_orihmap_gauhmap/'+str(i)+'_gauhmap.jpg', bbox_inches='tight', pad_inches=0)

        # combine gaussian heatmap with raw image
        img = video[..., i]
        img = cv2.resize(img, (224,224))
        hmap = hmaps[i, ...]
        hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
        comb = np.uint8(0.5*img[:,:,::-1]+0.5*hmap)
        cv2.imwrite('vis_res/comb_img_gauhmap_3/'+str(i)+'_test0.jpg', comb)


        

if __name__ == '__main__':
    main()