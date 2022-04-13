# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import argparse
import os
import logging

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
import numpy as np

from phoenix_datasets import PhoenixTVideoTextDataset, CSLDailyVideoTextDataset


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
    
    parser.add_argument('--dataset', type=str, default='2014', choices=['2014T', 'csl-daily'])
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--img_per_iter', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=2)

    args = parser.parse_args()
    return args


def create_dataloader(dset_name='2014', split='train', bsize=1):
    dset_dict = {'2014T': {'cls': PhoenixTVideoTextDataset, 'root': '../../data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T', 'mean': [0.5372,0.5273,0.5195], 'hmap_mean': []},
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
    
    if args.dataset == '2014T':
        path = os.path.join('/3tdisk/shared/rzuo/PHOENIX-2014-T', 'heatmaps_hrnet_mpii_9', args.split)
    elif args.dataset == 'csl-daily':
        path = os.path.join('/3tdisk/shared/rzuo/CSL-Daily', 'heatmaps_hrnet_mpii_9')

    if not os.path.exists(path):
        os.makedirs(path)
    
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(valid_loader), desc='[Generating heatmaps of {:s} of {:s}]'.format(args.split, args.dataset)):
            video = torch.cat(batch_data['video']).cuda()  #[T,3,256,256]
            len_video = batch_data['len_video'][0]
            video_id = batch_data['id'][0]
            fname = os.path.join(path, ''.join(video_id))
            
            #7 - thorax, 8 - upper neck
            #9 - head top, 
            #10 - r wrist, 11 - r elbow, 12 - r shoulder, 
            #13 - l shoulder, 14 - l elbow, 15 - l wrist
            heatmaps = model(video)
            heatmaps = heatmaps.detach().cpu().numpy()[:, 7:, ...]
            assert heatmaps.shape == (len_video,9,64,64)
            np.savez_compressed(fname+'.npz', heatmaps=heatmaps)


if __name__ == '__main__':
    main()