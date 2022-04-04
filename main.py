# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:11:16 2020

@author: Ronglai ZUO
"""

import argparse, yaml
import os
import logging
import torch as t

from phoenix_datasets import PhoenixVideoTextDataset, PhoenixTVideoTextDataset, PhoenixSIVideoTextDataset, PhoenixSI7VideoTextDataset, CSLVideoTextDataset, CSLDailyVideoTextDataset, TVBVideoTextDataset


def parse_args():
    p = argparse.ArgumentParser(description='SLR')
    p.add_argument('--config', type=str, default='config/default.yml')
    
    #model
    # p.add_argument('--model', type=str, default='lcasan', choices=['lcasan', 'CMA', 'fcn'])
    # p.add_argument('--batch_size', type=int, default=2)
    # p.add_argument('--vis_mod', type=str, default='vgg11', choices=['resnet18', 'vgg11', 'mb_v2', 'googlenet', 'cnn', 'dcn'])
    # p.add_argument('--seq_mod', type=str, default='transformer', choices=['tcn', 'transformer', 'gru', 'lstm', 'tcntr'])
    p.add_argument('--va', type=int, default=0, choices=[0,1], help='vac')
    p.add_argument('--ve', type=int, default=0, choices=[0,1], help='vac')
    p.add_argument('--alpha', type=float, default=25.0, help='factor of va loss')
    p.add_argument('--sema_cons', type=str, default=None, choices=[None, 'mask', 'drop', 'shuffle', 'multi_shuffle', 'drop_shuffle', 'drop_shuffle_insert', 'batch', 'frame', 'sequential', 'cosine'])
    p.add_argument('--sc_f', type=float, default=1.0, help='sentence embedding consistency factor')
    p.add_argument('--drop_ratio', type=float, default=1.0, help='drop_ratio for multi_shuffle, drop_shuffle, drop_shuffle_ins')
    p.add_argument('--pl', type=int, default=0, choices=[0,1], help='pseudo labeling')
    p.add_argument('--att_idx_lst', nargs='+', type=int, default=[], help='idx list of spatial attention blocks')
    p.add_argument('--spatial_att', type=str, default=None, choices=[None, 'dcn', 'dcn_nooff', 'ca', 'cbam'], help='number of coord att blocks')
    p.add_argument('--pool_type', type=str, default='avg', choices=['avg', 'gcnet-like', 'avg_softmax', 'cbam-like'], help='final pooling layer of vgg11')
    p.add_argument('--cbam_pool', type=str, default='max_softmax', choices=['max_avg', 'softmax', 'max_softmax'], help='channel pool type of CBAM')
    p.add_argument('--cbam_no_channel', type=int, default=1, choices=[0,1], help='whether use channel gate in CBAM')
    p.add_argument('--att_sup_type', type=str, default='first', choices=['first', 'all', 'res'])
    p.add_argument('--lr_factor', type=float, default=0.1, help='lr factor for some specicial modules')
    
    #score level
    p.add_argument('--D_std_gamma', nargs='+', type=float, default=[6.3, 1.4, 2.0], help='AAAI-2022 sub')
    p.add_argument('--mod_D', type=str, default=None, choices=[None, 'head_share', 'head_specific', 'nostat'])
    p.add_argument('--mod_src', type=str, default='Q', choices=['Q', 'K', 'ori_Q'])
    
    #feature level
    p.add_argument('--comb_conv', type=str, default=None, choices=[None, 'cascade', 'gate',\
                                                                   'cas_bef_san', 'cas_aft_ffn', 'add', 'linear'])
    
    #QK level
    p.add_argument('--qkv_context', nargs='+', type=int, default=[0,0,0])
    
    p.add_argument('--save_dir', type=str, default='./results')
    p.add_argument('--from_ckpt', type=int, default=0, choices=[0,1])
    p.add_argument('--max_num_epoch', type=int, default=60)
    p.add_argument('--num_pretrain_epoch', type=int, default=0)
    
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--setting', type=str, default='full', choices=['full', 'semi_ptf_50', 'semi_10', 'semi_20', 'semi_50', 'semi_100'])
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'vis_cam', 'vis_att', 'vis_vit', 'vis_fde'])
    p.add_argument('--test_split', type=str, default='test', choices=['test', 'dev', 'train'])  #maybe we want to test on training set
    
    #SLP
    # p.add_argument('--rec_model_file', type=str, default='./results/fcn_maskconv/ep79_step226880_wer25.70000.pkl')
    # p.add_argument('--sample_dir', type=str, default='./samples')
    
    #pose
    p.add_argument('--pose', type=str, default=None, choices=[None, 'filter', 'modality', 'super_att', \
                                                              'deform', 'deform_all', 'deform_mask', 'deform_patch', 'deform_and_mask', \
                                                                'vit_patch', 'prior'])
    p.add_argument('--pose_arg', nargs='+', type=float, default=[3,0.5], help='for heatmap filter: before which pooling layer, filter how many channels')
    p.add_argument('--pose_dim', type=int, default=0, help='for pose multi-modality')
    p.add_argument('--heatmap_num', type=int, default=3, choices=[7,3])
    p.add_argument('--heatmap_shape', type=int, nargs='+', default=[28], help='height of needed heatmap')
    p.add_argument('--heatmap_type', type=str, default='gaussian', choices=['origin', 'gaussian', 'norm'], help='origin means HRNet outputs')
    p.add_argument('--pose_f', type=float, default=1.0, help='pose factor of pose guided DCN')

    # feature disentange for SI
    p.add_argument('--fde', type=str, default=None, choices=[None, 's_and_c', 's_and_c_rev', 'distill', 'xvec', 'xvec_sim', 'xvec_sim_bank', \
                                                            'distill_share', 'similarity', 'dual_spat', 'dual_spat_xvec_sim',\
                                                            'dual_spat_xvec', 'adv', 'xvec_rev'])
    p.add_argument('--fde_loss_w', nargs='+', type=float, default=[0.0, 0.0, 0.0, 0.0], help='loss weights of fde')
    return p


if __name__ == '__main__':
    parser = parse_args()
    p = parser.parse_args()
    with open(p.config, 'r') as f:
        default_args = yaml.load(f, Loader=yaml.FullLoader)
    parser.set_defaults(**default_args)
    args = parser.parse_args()
    config = vars(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    from utils.utils import setup_seed, init_logging, get_param_count
    setup_seed(args.seed)
    
    from base import TrainingManager
    from semi import TrainingManagerSemi
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.mode == 'train':
        init_logging(os.path.join(args.save_dir, '{:s}_log.txt'.format('Training Starts')))
    elif args.mode == 'test':
        init_logging(os.path.join(args.save_dir, '{:s}_log.txt'.format('Testing Starts')))
    logging.info('Use random seed: {:d}'.format(args.seed))
    for key in config.keys():
        logging.info(key+': '+str(config[key]))
    
    # save important scripts into save_dir
    if args.mode == 'train':
        os.system("cp {:s} {:s}".format(args.config, args.save_dir))
        os.system("cp base.py {:s}".format(args.save_dir))
        os.system("cp phoenix_datasets/corpora.py {:s}".format(args.save_dir))
        os.system("cp model.py {:s}".format(args.save_dir))
        os.system("cp tfmer/transformer_layers.py {:s}".format(args.save_dir))
        # os.system("cp modules/vit.py {:s}".format(args.save_dir))
        os.system("cp modules/fde.py {:s}".format(args.save_dir))

    dset_dict = {'2014': {'cls': PhoenixVideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-multisigner', 'mean': [0,0,0]},
                 '2014T': {'cls': PhoenixTVideoTextDataset, 'root': '../../data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T', 'mean': [0,0,0]},
                 '2014SI': {'cls': PhoenixSIVideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-signerindependent-SI5', 'mean': [0,0,0]},
                 '2014SI7': {'cls': PhoenixSI7VideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-signerindependent-SI5', 'mean': [0,0,0]},
                 'csl1': {'cls': CSLVideoTextDataset, 'root': ('../../data/ustc-csl', 'split_1.txt'), 'mean': [0,0,0]},
                 'csl2': {'cls': CSLVideoTextDataset, 'root': ('../../data/ustc-csl', 'split_2.txt'), 'mean': [0,0,0]},
                 'csl-daily': {'cls': CSLDailyVideoTextDataset, 'root': '../../data/csl-daily', 'mean': [0,0,0]},
                 'tvb': {'cls': TVBVideoTextDataset, 'root': '/6tdisk/shared/tvb', 'mean': [0,0,0]}}
    dset_dict = dset_dict[args.data['dataset']]
    dtrain = dset_dict['cls'](args=args.data,
                              root=dset_dict['root'],
                              normalized_mean=dset_dict['mean'],
                              split='train',
                              use_random=False)
    
    vocab = list(dtrain.vocab.table.keys())
    logging.info('len vocab: {:d}, len dtrain: {:d}'.format(len(vocab), len(dtrain)))
    del dtrain
    
    if args.setting == 'full':
        training_manager = TrainingManager(args, vocab)
    elif 'semi' in args.setting:
        training_manager = TrainingManagerSemi(args, vocab)
    
    if args.mode == 'train':
        num_param = get_param_count(training_manager.model)
        logging.info('Number of Parameters: {:.6f}'.format(num_param))
        if 'ptf' in args.setting:
            training_manager.train_ptf()
        else:
            training_manager.train()

        # training_manager.validate(0, 0, 'dev')  #for measure inference speed
        
        training_manager.args.mode = 'test'
        for fname in os.listdir(args.save_dir):
            if 'pkl' in fname and 'ep' in fname:
                model_file = os.path.join(args.save_dir, fname)
                training_manager.validate(args.max_num_epoch, 0, 'test', model_file)
    
    elif args.mode == 'test':
        for fname in os.listdir(args.save_dir):
            if 'pkl' in fname and 'ep' in fname:
                model_file = os.path.join(args.save_dir, fname)
                training_manager.validate(args.max_num_epoch, 0, args.test_split, model_file)
    
    # elif args.mode == 'vis_cam':
    #     for fname in os.listdir(args.save_dir):
    #         if 'pkl' in fname and 'ep' in fname:
    #             model_file = os.path.join(args.save_dir, fname)
    #             training_manager.vis_cam(model_file)
    
    elif args.mode == 'vis_att':
        for fname in os.listdir(args.save_dir):
            if 'pkl' in fname and 'ep' in fname:
                model_file = os.path.join(args.save_dir, fname)
                training_manager.vis_attention(model_file)
                
    elif args.mode == 'vis_vit':
        for fname in os.listdir(args.save_dir):
            if 'pkl' in fname and 'ep' in fname:
                model_file = os.path.join(args.save_dir, fname)
                training_manager.vis_vit(model_file)

    elif args.mode == 'vis_fde':
        for fname in os.listdir(args.save_dir):
            if 'pkl' in fname and 'ep' in fname:
                model_file = os.path.join(args.save_dir, fname)
                training_manager.vis_fde(model_file)
