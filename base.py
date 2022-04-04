# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:33:01 2020

@author: Ronglai ZUO
Script for base class
"""
from pickletools import optimize
import torch as t; t.backends.cudnn.deterministic = True#; t.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, pickle
import logging
import uuid
import arpa

from model import SLRModel, CMA
from modules.dcn import gen_mask_from_pose
from phoenix_datasets import PhoenixVideoTextDataset, PhoenixTVideoTextDataset, PhoenixSIVideoTextDataset, PhoenixSI7VideoTextDataset, CSLVideoTextDataset, CSLDailyVideoTextDataset, TVBVideoTextDataset
from utils.metric import get_wer_delsubins
from utils.utils import update_dict, worker_init_fn, LossManager, ModelManager, freeze_params, unfreeze_params, record_loss
from utils.figure import gen_att_map
from evaluation_relaxation.phoenix_eval import get_phoenix_wer
from ctcdecode import CTCBeamDecoder
from modules.criterion import SeqKD, SeqJSD, ConfLoss, LabelSmoothCE, FocalLoss
from modules.fde import SignerClassifier
# from modules.searcher import CTC_decoder

from itertools import groupby
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
from collections import defaultdict

# import matplotlib#; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class TrainingManager(object):
    def __init__(self, args, vocab):
        self.args = args
        self.args_data, self.args_model, self.args_dcn, self.args_tf, self.args_opt, self.args_lr_sch = \
            args.data, args.model, args.dcn, args.transformer, args.optimizer, args.lr_scheduler
        self.vocab = vocab  #for language model
        self.voc_size = len(vocab)
        self.blank_id = vocab.index('blank')
        # assert len(vocab) in [1233, 1234, 1119, 180, 28]
        assert self.blank_id == self.voc_size-1
        
        self.model = None
        # if self.args_model['name'] == 'lcsa' or self.args_model['name'] == 'fcn':
        pose_arg = [args.pose]
        pose_arg.extend(args.pose_arg)
        self.model = SLRModel(args_model=self.args_model,
                                args_tf=self.args_tf,
                                D_std_gamma=args.D_std_gamma,
                                mod_D=args.mod_D,
                                mod_src=args.mod_src,
                                comb_conv=args.comb_conv,
                                qkv_context=args.qkv_context,
                                gls_voc_size=self.voc_size, 
                                pose_arg=pose_arg,
                                pose_dim=args.pose_dim,
                                dcn_ver=self.args_dcn['ver'],
                                att_idx_lst=args.att_idx_lst,
                                spatial_att=args.spatial_att,
                                pool_type=args.pool_type,
                                cbam_no_channel=bool(args.cbam_no_channel),
                                cbam_pool=args.cbam_pool,
                                ve=bool(args.ve) or bool(args.va),
                                sema_cons=args.sema_cons,
                                drop_ratio=args.drop_ratio,
                                fde=args.fde,
                                num_signers=8 if '2014SI' in args.data['dataset'] else 9
                                )
        # elif self.args_model['name'] == 'CMA':
        # self.model = CMA(gls_voc_size=self.voc_size)
        
        self.optimizer = self._create_optimizer(self.model)
        self.lr_scheduler = self._create_lr_scheduler(self.optimizer)
        self.model_D = self.optimizer_D = self.lr_scheduler_D = None
        self.criterion = nn.CTCLoss(self.blank_id, zero_infinity=True).cuda()
        if args.ve:
            self.ve_crit = nn.CTCLoss(self.blank_id, zero_infinity=True).cuda()
        if args.va:
            self.va_crit = SeqKD(T=8, blank_id=self.blank_id).cuda()

        if args.pose is not None and ('deform' in args.pose or args.pose in ['super_att', 'vit_patch']):
            # self.pose_crit = nn.SmoothL1Loss().cuda()
            self.pose_crit = nn.MSELoss().cuda()
        
        if args.sema_cons == 'cosine':
            self.sema_crit = nn.CosineEmbeddingLoss().cuda()
        elif args.sema_cons is not None:
            self.sema_crit = nn.TripletMarginWithDistanceLoss(distance_function=lambda x,y: 1.0-F.cosine_similarity(x,y), margin=2.0).cuda()
        
        if args.fde is not None:
            # fde signer classifier loss
            # ce_w = t.FloatTensor([1475,49,470,836,30,647,704,165]).cuda()
            # ce_w = ce_w.sum() - ce_w
            # ce_w /= ce_w.sum()
            self.fde_cls_crit = nn.CrossEntropyLoss().cuda()
            # self.fde_cls_crit = FocalLoss(gamma=1.0, weight=None).cuda()
            # self.fde_cls_crit = LabelSmoothCE(lb_smooth=0.1).cuda()
            self.fde_cam_crit = nn.MSELoss().cuda()
            # self.fde_cam_crit = SeqJSD(T=8, blank_id=self.blank_id).cuda()
            self.fde_rkl_crit = SeqJSD(T=8, blank_id=self.blank_id).cuda()  #reversed knowledge distillation
            if 'adv' in args.fde:
                self.model_D = SignerClassifier(512, 8).cuda()
                self.optimizer_D = self._create_optimizer(self.model_D)
                self.lr_scheduler_D = self._create_lr_scheduler(self.optimizer_D)
                self.fde_conf_crit = ConfLoss().cuda()
        
        ctc_decoder_vocab = [chr(x) for x in range(20000, 20000 + self.voc_size)]
        self.ctc_decoder = CTCBeamDecoder(ctc_decoder_vocab,
                                          beam_width=args.beam_size,
                                          blank_id=self.blank_id,
                                          num_processes=5
                                          )

        self.decoded_dict = {}
        self.signer_emb_bank = {}
        if self.args.fde == 'xvec_sim_bank':
            for i in range(8):
                self.signer_emb_bank[str(i)] = t.zeros(self.args_model['emb_size']).cuda()
                self.signer_emb_bank['num_'+str(i)] = 0
        self.dset_dict = {'2014': {'cls': PhoenixVideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-multisigner', 'mean': [0.5372,0.5273,0.5195], 'hmap_mean': [0.0236, 0.0250, 0.0164, 0.0283, 0.0305, 0.0240, 0.0564]},
                          '2014T': {'cls': PhoenixTVideoTextDataset, 'root': '../../data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T', 'mean': [0.5372,0.5273,0.5195], 'hmap_mean': [0,0,0,0,0,0,0]},
                          '2014SI': {'cls': PhoenixSIVideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-signerindependent-SI5', 'mean': [0.5405, 0.5306, 0.5235], 'hmap_mean': [0,0,0,0,0,0,0]},
                          '2014SI7': {'cls': PhoenixSI7VideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-signerindependent-SI5', 'mean': [0.5400, 0.5295, 0.5225], 'hmap_mean': [0,0,0,0,0,0,0]},
                          'csl1': {'cls': CSLVideoTextDataset, 'root': ('../../data/ustc-csl', 'split_1.txt'), 'mean': [0.5827, 0.5742, 0.5768], 'hmap_mean': [0,0,0,0,0,0,0]},
                          'csl2': {'cls': CSLVideoTextDataset, 'root': ('../../data/ustc-csl', 'split_2.txt'), 'mean': [0.5827, 0.5742, 0.5768], 'hmap_mean': [0,0,0,0,0,0,0]},
                          'csl-daily': {'cls': CSLDailyVideoTextDataset, 'root': '../../data/csl-daily', 'mean': [0.6891, 0.6680, 0.6409], 'hmap_mean': [0,0,0,0,0,0,0]},
                          'tvb': {'cls': TVBVideoTextDataset, 'root': '/6tdisk/shared/tvb', 'mean': [0.4874, 0.5383, 0.5366], 'hmap_mean': [0,0,0,0,0,0,0]}}
        if args.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir=args.save_dir + "/tensorboard/")
    
    
    def _create_optimizer(self, model):
        #filter for semantics extractor
        if self.args_opt['name'] == 'adam':
            return t.optim.Adam([{'params': [p for n, p in model.named_parameters() if 'sema_ext' not in n]},
                                {'params': [p for n, p in model.named_parameters() if 'sema_ext' in n], 'lr': self.args_opt['lr']*self.args.lr_factor}],
                                self.args_opt['lr'], self.args_opt['betas'], self.args_opt['weight_decay'])
        elif self.args_opt['name'] == 'sgd':
            return t.optim.SGD(model.parameters(), self.args_opt['lr'], self.args_opt['momentum'], self.args_opt['weight_decay'])
            
            
    def _create_lr_scheduler(self, optimizer):
        if self.args_lr_sch['name'] == 'plateau':
            return t.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                verbose=False,
                threshold_mode="abs",
                factor=self.args_lr_sch['decrease_factor'],
                patience=self.args_lr_sch['patience'],  #6 eval steps!
            )
        elif self.args_lr_sch['name'] == 'step':
            return t.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                            step_size=2,  #2 epochs!
                                            gamma=self.args_lr_sch['decrease_factor'])
        elif self.args_lr_sch['name'] == 'mstep':
            return t.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                    # milestones=[20,30,40,50,60,65,70,75],
                                                    # milestones=[20,30,40,45,50,55],
                                                    # milestones=[15,25,30,35,40,45],  #CVPR22
                                                    milestones=[15,25,30,35,40,45,50],  #TMM22
                                                    gamma=self.args_lr_sch['decrease_factor'])
            
            
    def create_dataloader(self, split='train', bsize=4, use_random=None):
        if use_random is None:
            if split == 'train':
                use_random = True
            else:
                use_random = False
        
        # heatmap_shape=[64,64]
        # if self.args.pose == 'filter':
        #     #compute heatmap size
        #     if self.args.pose_arg[0] > 0:
        #         heatmap_shape = [int(224/2**(self.args.pose_arg[0]-1)), int(224/2**(self.args.pose_arg[0]-1))]
        #     else:
        #         heatmap_shape = [224,224]

        dset_dict = self.dset_dict[self.args_data['dataset']]
        dset_cls = dset_dict['cls']
        dataset = dset_cls(
                # your path to this folder, download it from official website first.
                args=self.args_data,
                root=dset_dict['root'],
                normalized_mean=dset_dict['mean'],
                split=split,
                use_random=use_random if 'vis' not in self.args.mode else False,
                pose=self.args.pose if split=='train' or self.args.mode=='vis_att' or self.args.pose=='filter' or self.args.pose == 'vit_patch' else None,
                heatmap_shape=self.args.heatmap_shape,
                heatmap_num=self.args.heatmap_num,
                heatmap_mean=dset_dict['hmap_mean'],
                heatmap_type=self.args.heatmap_type
                )
        
        spler = None
        if split == 'train':
            self.len_dtrain = len(dataset)
            if 'semi' in self.args.setting:
                ratio = int(self.args.setting.split('_')[-1]) / 100
                train_idx = np.arange(self.len_train)
                spler = sampler.SubsetRandomSampler(train_idx[:int(self.len_dtrain*ratio)])
        
        dataloader = DataLoader(dataset, 
                                bsize,
                                shuffle=True if split=='train' and 'vis' not in self.args.mode else False,
                                sampler=spler,
                                num_workers=8,
                                # worker_init_fn=worker_init_fn if self.args.batch_size==4 else None,
                                collate_fn=dataset.collate_fn,
                                drop_last=True)
        return dataloader

    
    def eval_batch(self, batch_data, need_att=False):
        with t.no_grad():
            batch_size = len(batch_data['video'])
            video = t.cat(batch_data['video']).cuda()
            len_video = batch_data['len_video'].cuda()
            label = batch_data['label'].cuda()
            len_label = batch_data['len_label'].cuda()
            video_id = batch_data['id']
            coord = None
            if self.args.pose is not None:
                coord = t.cat(batch_data['coord']).cuda()
            # heatmap = [None]
            # if self.args.pose in ['filter', 'modality']:
            #     heatmap = []
            #     for hmap in zip(*batch_data['heatmap']):
            #         heatmap.append(t.cat(hmap).cuda()) 
            
            self.model.eval()
            op_dict = self.model(video, len_video, coord=coord, return_att=need_att)
            gls_logits, len_video, plot_lst, semantics = op_dict['gls_logits'], op_dict['len_video'], op_dict['plot'], op_dict['semantics']
            
            #compute validaiton loss
            gls_prob = F.log_softmax(gls_logits, -1)
            gls_prob = gls_prob.permute(1,0,2)
            val_loss = self.criterion(gls_prob, label, len_video, len_label)
            
            #ctc decode
            gls_prob = F.softmax(gls_logits, dim=-1)
            pred_seq, beam_scores, _, out_seq_len = self.ctc_decoder.decode(gls_prob, len_video)
            # else:
            #     pool = Pool(5)
            #     decode_res = pool.map(self.ctc_decoder.decode, gls_scores.cpu().numpy(), len_video.cpu().numpy())
            #     dec_hyp = []
            #     for res in decode_res:
            #         dec_hyp.append([x[0] for x in groupby(res[0][0][:res[3][0]].tolist())])
            
            #metrics evaluation: wer
            assert pred_seq.shape[0] == batch_size
            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            start = 0
            hyp_lst = []
            for i, length in enumerate(len_label):
                end = start + length
                ref = label[start:end].tolist()
                hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                # hyp_lst.append(hyp)
                # hyp = self.get_hyp_after_lm(pred_seq[i], beam_scores[i], out_seq_len[i])
                
                id = ''.join(video_id[i])
                self.decoded_dict[id] = hyp
                
                correct += int(ref == hyp)
                colors = ['blue', 'brown', 'red', 'green', 'gold', 'pink', 'cyan', 'lime', 'purple', 'tomato', 'yellowgreen', 'maroon', 'black']
                # if correct and label.shape[0] <= len(colors):
                #     print(id)
                #     D = plot_lst[1][0, :, :, 0].cpu().numpy()
                #     x = np.arange(0, D.shape[-1])*2
                #     fig, axes = plt.subplots(9, 1, sharex=True)
                #     for i in range(8):
                #         axes[i].plot(x, D[i], lw=1, color=colors[i])
                #         axes[i].spines['right'].set_visible(False)
                #         axes[i].spines['top'].set_visible(False)
                #     fig.add_subplot(111, frameon=False)
                #     # hide tick and tick label of the big axis
                #     plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
                #     plt.xlabel("time step")
                #     plt.ylabel("window size")
                #     for k in range(label.shape[0]):
                #         l = label.cpu()[k]
                #         y = gls_prob[0, :, l].cpu().numpy()
                #         axes[-1].plot(x, y, c=colors[k], lw=1)
                #     axes[-1].set_ylabel('prob')
                #     axes[-1].spines['right'].set_visible(False)
                #     axes[-1].spines['top'].set_visible(False)
                #     plt.savefig(self.args.save_dir + '/' + id + '.jpg')

                # if correct and label.shape[0] <= len(colors) and id in self.head_share_ori.keys():
                #     # self.wsize_dict[id] = plot_lst[1][0,0,:,0].cpu().numpy()
                #     # self.wsize_dict[id] = gls_scores[0, :, label].cpu().numpy()
                #     max_len = 999
                #     if id == '29November_2011_Tuesday_tagesschau_default-10':
                #         print('yes')
                #         max_len = 31
                #     D = plot_lst[1][0, 0, :max_len, 0].cpu().numpy()
                #     D_ori = self.head_share_ori[id][:max_len]
                #     x = np.arange(0, D.shape[0])*2
                #     plt.style.use('default')
                #     fig = plt.figure(constrained_layout=True)
                #     gs = fig.add_gridspec(2, 2)
                #     ax1 = fig.add_subplot(gs[0, :])
                #     ax1.plot(x, D, lw=1, label='Q^L')
                #     ax1.plot(x, D_ori, lw=1, label='Q')
                #     ax1.set_ylabel('window size')
                #     ax1.legend()
                #     ax1.spines['right'].set_visible(False)
                #     ax1.spines['top'].set_visible(False)
                    
                #     ax2 = fig.add_subplot(gs[1, :])
                #     for k in range(label.shape[0]):
                #         l = label.cpu()[k]
                #         y = gls_prob[0, :max_len, l].cpu().numpy()
                #         ax2.plot(x, y, c=colors[k], lw=1)
                #     ax2.set_xlabel('time step')
                #     ax2.set_ylabel('probability')
                #     ax2.spines['right'].set_visible(False)
                #     ax2.spines['top'].set_visible(False)
                    
                #     idx, _ = find_peaks(-D, distance=2.5)
                #     ax = fig.add_subplot(gs[:, :], sharex = ax1)
                #     ax.patch.set_alpha(0)
                #     ax.axis("off")
                #     for ele in idx:
                #         ax.axvline(2*ele, c='grey', ls='--', lw=0.8)
                #     plt.savefig(self.args.save_dir + '/' + id + '.pdf')

                # distance of sentence embedding
                # anc, pos = semantics
                # neg = pos.index_select(0, t.tensor([1,0]).cuda())
                # dist_pos = (1.0-F.cosine_similarity(anc, pos)).detach().cpu().numpy()  #[B] or [B,T]
                # dist_neg = (1.0-F.cosine_similarity(anc, neg)).detach().cpu().numpy()
                # if len(dist_pos.shape) == 2:
                #     dist_pos = dist_pos.mean(axis=-1)
                #     dist_neg = dist_neg.mean(axis=-1)
                # dist = np.concatenate([dist_pos, dist_neg], axis=0)
                # if i == 0:
                #     self.decoded_dict[id] = np.array([dist_pos[0], dist_neg[0]])
                #     id = ''.join(video_id[1])
                #     self.decoded_dict[id] = np.array([dist_pos[1], dist_neg[1]])

                err = get_wer_delsubins(ref, hyp, debug=False, vocab=self.vocab, save_dir=self.args.save_dir, id=id)
                err_delsubins += np.array(err)
                count += 1
                start = end
                
            assert end == label.size(0)
        
        return err_delsubins, correct, count, {'loss': val_loss.item(), 
                                               # 'gate': t.mean(plot['gate'][0,:len_video.cpu()[0],:]).cpu().numpy() if self.args.comb_conv == 'gate' else None,
                                               'att_1': plot_lst[0].mean(dim=(-2,-1)).cpu() if need_att and self.args_model['seq_mod']=='transformer' and self.args_tf['pe']=='rpe_gau' else None,
                                               'att_2': plot_lst[1].mean(dim=(-2,-1)).cpu() if need_att and self.args_model['seq_mod']=='transformer' and self.args_tf['pe']=='rpe_gau' else None,
                                               'ratio': (len_video/len_label).cpu(),
                                               'hyp_lst': hyp_lst,
                                               'miscell': None}
    
    
    def train_batch(self, batch_data, epoch):
        video = t.cat(batch_data['video']).cuda()
        len_of_video = batch_data['len_video'].cuda()
        label = batch_data['label'].cuda()
        len_label = batch_data['len_label'].cuda()
        signer = None
        if self.args.fde is not None:
            signer = batch_data['signer']
            if 'xvec' in self.args.fde:
                signer = t.tensor(signer).cuda()
            else:
                for i in range(len(signer)):
                    signer[i] = t.tensor(signer[i]).expand(len_of_video[i])
                signer = t.cat(signer, dim=0).cuda()
        coord, heatmap = None, None
        if self.args.pose in ['deform', 'deform_all', 'deform_patch', 'vit_patch']:
            coord = t.cat(batch_data['coord']).cuda()
        # elif self.args.pose == 'modality':
        #     heatmap = t.cat(batch_data['coord']).cuda()
        elif self.args.pose == 'deform_and_mask':
            coord = t.cat(batch_data['coord']).cuda()
            heatmap = []
            for hmap in zip(*batch_data['heatmap']):
                heatmap.append(t.cat(hmap).cuda())
        elif self.args.pose is not None:
            heatmap = []
            for hmap in zip(*batch_data['heatmap']):
                heatmap.append(t.cat(hmap).cuda())
        if self.args.heatmap_type == 'norm':
            heatmap_norm = []
            for hmap in zip(*batch_data['heatmap_norm']):
                heatmap_norm.append(t.cat(hmap).cuda())

        self.model.train()
        op_dict = self.model(video, len_of_video, coord=coord, signer=signer, signer_emb_bank=self.signer_emb_bank)
        gls_logits, vis_logits, len_video, offset_lst, mask_lst, semantics = \
            op_dict['gls_logits'], op_dict['vis_logits'], op_dict['len_video'], op_dict['offset'], op_dict['spat_att'], op_dict['semantics']
        gls_prob = gls_logits.log_softmax(-1)
        gls_prob = gls_prob.permute(1,0,2)
        
        loss = self.criterion(gls_prob, label, len_video, len_label)
        loss = loss.mean()
        
        loss_pose = t.tensor(0.0).cuda()
        if self.args.pose in ['deform', 'deform_patch', 'deform_and_mask']:
            offset = offset_lst[0]
            T, C, H, H = offset.shape
            assert C == 18
            # select the offset of the center of the 3*3 kernel
            sel_mask = t.zeros(1, C, 1, 1).bool().cuda()
            sel_mask[0, (4,13), 0, 0] = True
            cen_offset = offset.masked_select(sel_mask).view(T,2,H,H)
            grid_x, grid_y = t.meshgrid(t.arange(H), t.arange(H))
            init_coords = t.stack([grid_x, grid_y], dim=0).float().cuda()
            
            # coords after shift
            cen_offset = cen_offset.add(init_coords).div(H-1)  #[T,2,H,H]
            
            if self.args.pose in ['deform', 'deform_and_mask']:
                cen_offset = cen_offset.view(T,2,-1)  #[T,2,HH]
                loss_pose_lst = []
                for c in coord.split(1, dim=1):
                    c = c.permute(0,2,1).expand_as(cen_offset)  #[T,2,HH]
                    loss_pose_lst.append(self.pose_crit(cen_offset, c))
                loss_pose += min(loss_pose_lst)
            
            elif self.args.pose == 'deform_patch':
                mask_h, mask_lw, mask_rw = t.zeros(1,1,H,H).bool().cuda(), t.zeros(1,1,H,H).bool().cuda(), t.zeros(1,1,H,H).bool().cuda()
                mask_h[0, 0, 0:H//2, :] = True
                mask_rw[0, 0, H//2:, 0:H//2] = True
                mask_lw[0, 0, H//2:, H//2:] = True
                
                patch_offset_lst = []
                coord_lst = []
                for c, mask in zip(coord.split(1, dim=1), [mask_h,mask_lw,mask_rw]):
                    patch_offset = cen_offset.masked_select(mask).view(T,2,-1)
                    patch_offset_lst.append(patch_offset)
                    coord_lst.append(c.permute(0,2,1).expand_as(patch_offset))
                
                # stack
                patch_offset = t.cat(patch_offset_lst, dim=2)
                c = t.cat(coord_lst, dim=2)
                loss_pose += self.pose_crit(patch_offset, c)
                
        elif self.args.pose == 'deform_all':
            # for i in range(2):
            offset = offset_lst[0]
            T, C, H, H = offset.shape
            assert C == 18

            grid_x, grid_y = t.meshgrid(t.arange(H), t.arange(H))
            grid_x, grid_y = grid_x.repeat(9,1,1), grid_y.repeat(9,1,1)
            init_coords = t.cat([grid_x, grid_y], dim=0).float().cuda()  #[18,H,H]
            init_offset = t.tensor([-1,-1,-1,0,0,0,1,1,1,-1,0,1,-1,0,1,-1,0,1]).cuda().view(18,1,1)
            init_coords += init_offset
            
            # coords after shift
            offset = offset.add(init_coords).div(H-1)  #[T,18,H,H]
            offset = offset.view(T,C,-1)  #[T,18,HH]
            
            loss_pose_lst = []
            for c in coord.split(1, dim=1):
                x, y = c.split(1, dim=2)
                x, y = x.permute(0,2,1).expand(T,C//2,1), y.permute(0,2,1).expand(T,C//2,1)  #[T,9,1]
                co = t.cat([x,y], dim=1).expand_as(offset)  #[T,18,HH]
                loss_pose_lst.append(self.pose_crit(offset, co))
            loss_pose += min(loss_pose_lst)
            
        if self.args.pose in ['deform_mask', 'deform_and_mask']:
            mask = mask_lst[0]
            mask_label = gen_mask_from_pose(offset_lst[0], heatmap[0]).detach()
            loss_pose += self.pose_crit(mask, mask_label)
            
        if self.args.pose == 'super_att' and self.args.fde is None:
            if self.args_model['vis_mod'] == 'mb_v2':
                idx = 1
            elif self.args_model['vis_mod'] in ['vgg11', 'cnn']:
                idx = 0
            elif self.args_model['vis_mod'] == 'resnet18':
                idx = 3
            
            #supervise the first attention block/all blocks/one block for each resolution/all blocks on a single resolution
            if self.args.att_sup_type == 'first':
                masks = mask_lst[idx:idx+1]
                offsets = offset_lst[idx:idx+1]
                # if self.args.spatial_att == 'cbam':
                #     masks = mask_lst[idx+1:idx+2]
                #     offsets = offset_lst[idx+1:idx+2]
            elif self.args.att_sup_type == 'all':
                masks = mask_lst[idx:]
                offsets = offset_lst[idx:]
            elif self.args.att_sup_type == 'res':
                # masks = mask_lst[idx::2]
                # offsets = offset_lst[idx::2]
                # if self.args.spatial_att == 'cbam':
                masks = mask_lst[idx:idx+2]
                offsets = offset_lst[idx:idx+2]
            
            upsample = False
            if self.args.spatial_att in ['dcn', 'dcn_nooff']:
                for mask, offset in zip(masks, offsets):
                    mask_label = gen_mask_from_pose(offset, heatmap[self.args.heatmap_shape.index(mask.shape[-1])]).detach()
                    loss_pose += self.pose_crit(mask, mask_label)
            elif self.args.spatial_att == 'ca':
                for mask in masks:
                    mask_label = heatmap[self.args.heatmap_shape.index(mask.shape[-1])]
                    loss_pose += self.pose_crit(mask.mean(dim=1, keepdim=True), mask_label)
            elif self.args.spatial_att == 'cbam':
                for mask in masks:
                    if upsample:
                        mask = F.upsample(mask, size=self.args.heatmap_shape[0])
                    mask_label = heatmap[self.args.heatmap_shape.index(mask.shape[-1])]
                    loss_pose += self.pose_crit(mask, mask_label)
            loss_pose /= len(masks)
        
        elif self.args.pose == 'vit_patch' and offset_lst[0] is not None:
            loss_pose = self.pose_crit(t.stack(offset_lst, dim=0), coord.expand(len(offset_lst), -1, -1, -1))
            
        loss += self.args.pose_f * loss_pose
        
        # loss of semantic consistency
        loss_sc = t.tensor(0.0).cuda()
        if self.args.sema_cons == 'cosine':
            anc, pos = semantics
            loss_sc = self.sema_crit(anc.detach(), pos, t.tensor([1,1]).cuda())
        elif self.args.sema_cons is not None:
            if self.args.sema_cons in ['batch', 'frame']:
                anc, pos = semantics
                neg = pos.index_select(0, t.tensor([1,0]).cuda())
            elif self.args.sema_cons == 'sequential':
                pos, anc = semantics
                neg = pos.index_select(0, t.tensor([1,0]).cuda())
            else:
                anc, pos, neg = semantics
            loss_sc = self.sema_crit(anc.detach(), pos, neg)
        loss += self.args.sc_f * loss_sc
        
        if self.args.ve:
            ve_vis_prob = vis_logits.log_softmax(-1).permute(1,0,2)
            loss_ve = self.ve_crit(ve_vis_prob, label, len_of_video, len_label)  #ctc loss
            loss += loss_ve
        if self.args.va:
            loss_va = self.va_crit(vis_logits, gls_logits.detach()) 
            loss += self.args.alpha * loss_va
        
        # feature disentangle
        loss_signer = loss_cam = loss_ch_cam = loss_conf = t.tensor(0.0).cuda()
        w_signer, w_cam, w_ch_cam, w_rkl = self.args.fde_loss_w
        if self.args.fde is not None:
            sg, sg2, cam = op_dict['cam']
            cg, ch_cam = op_dict['ch_cam']
            # signer classification loss
            if w_signer > 0:
                if 'adv' in self.args.fde:
                    freeze_params(self.model_D)
                    signer_emb = op_dict['signer_emb']
                    signer_logits = self.model_D(signer_emb)
                    loss_conf = self.fde_conf_crit(signer_logits)
                    loss += w_signer * loss_conf
                else:
                    loss_signer = self.fde_cls_crit(op_dict['signer_logits'], signer)
                    loss += w_signer * loss_signer
            # CAM guidance loss
            if w_cam > 0:
                # if 'sim' in self.args.fde:
                #     if epoch >= 15:
                #         loss_cam = self.fde_cam_crit(sg, cam.detach())
                #         loss += -w_cam * loss_cam
                # else:
                sg_label = heatmap[0]
                # if self.args.fde not in ['s_and_c', 'xvec', 'dual_spat', 'dual_spat_xvec', 'dual_spat_xvec_sim', 'adv'] and epoch >= 15:
                #     sg_label *= (1-cam)
                loss_cam = self.fde_cam_crit(sg, sg_label.detach())
                loss += w_cam * loss_cam
            # ch-CAM guidance loss
            if w_ch_cam > 0 and epoch >= 15:
                # loss_ch_cam = (self.fde_cam_crit(cg, (1-ch_cam).detach()) + self.fde_cam_crit((1-ch_cam), cg.detach()))/2
                if 'dual_spat' in self.args.fde:
                    loss_ch_cam = self.fde_cam_crit(sg2, (1-cam).detach())
                else:
                    loss_ch_cam = self.fde_cam_crit(cg, (1-ch_cam).detach())
                loss += w_ch_cam * loss_ch_cam
            if self.args.fde in ['distill', 'distill_share'] and epoch >= 0:
                # loss_rkl = (self.fde_rkl_crit(vis_logits, gls_logits.detach()) + self.fde_rkl_crit(gls_logits, vis_logits.detach()))/2
                loss_rkl = self.fde_rkl_crit(vis_logits, gls_logits.detach())
                loss += -w_rkl * loss_rkl

        loss.backward()
        self.optimizer.step()
        if self.args.fde is not None and 'adv' in self.args.fde:
            unfreeze_params(self.model_D)
            loss_signer = self.fde_cls_crit(self.model_D(signer_emb.detach()), signer)
            (w_signer * loss_signer).backward()
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
        self.optimizer.zero_grad()

        return {'loss': loss.item(), 
                'loss_signer': loss_signer.item(), 'loss_cam': loss_cam.item(), 'loss_ch_cam': loss_ch_cam.item(), 'loss_conf': loss_conf.item(),
                'loss_pose': loss_pose.item(), 'loss_sc': loss_sc.item()}
    
    
    def train(self):
        dtrain = self.create_dataloader(split='train', bsize=self.args_model['batch_size'])
        loss_manager = LossManager(print_step=100)
        self.model_manager = ModelManager(max_num_models=5)  #only save the best 3 models
        
        self.model.cuda()
        max_num_epoch = self.args.max_num_epoch
        global_step = 0
        last_status = {'loss': -1., 'loss_trip': -1.}
        start_epoch = 0
        
        if self.args.from_ckpt:
            ckpt_file = os.path.join(self.args.save_dir, 'latest.pkl')
            print('loading from {:s}'.format(ckpt_file))
            saved_dict = t.load(ckpt_file)
            self.model.load_state_dict(saved_dict['mainstream'])
            self.optimizer.load_state_dict(saved_dict['optimizer'])
            self.lr_scheduler.load_state_dict(saved_dict['lr_scheduler'])
            if self.model_D is not None:
                self.model_D.load_state_dict(saved_dict['model_D'])
                self.optimizer_D.load_state_dict(saved_dict['optimizer_D'])
                self.lr_scheduler_D.load_state_dict(saved_dict['lr_scheduler_D'])
            start_epoch = saved_dict['epoch']+1
            global_step = self.len_dtrain // self.args_model['batch_size'] * start_epoch
            t.manual_seed(self.args.seed+start_epoch*3)  #change dataloader order
            dtrain = self.create_dataloader(split='train', bsize=self.args_model['batch_size'])
        
        for epoch in range(start_epoch, max_num_epoch):
            #*********************Training*******************
            epoch_loss = defaultdict(list)
            for i, batch_data in tqdm(enumerate(dtrain), desc='[Training, epoch {:d}]'.format(epoch)):
                global_step += 1
                loss_dict = self.train_batch(batch_data, epoch)
                loss_manager.update(loss_dict, global_step)
                record_loss(loss_dict, epoch_loss)

                if self.args_lr_sch['patience'] == 6 and i == self.len_dtrain//(self.args_model['batch_size']*2) and 'step' not in self.args_lr_sch['name']:  
                    #half of the epoch
                    self.validate(epoch, global_step)
            
            logging.info('Epoch: {:d}, loss: {:.3f} -> {:.3f}'.format(epoch, last_status['loss'], np.mean(epoch_loss['loss'])))
            last_status['loss'] = np.mean(epoch_loss['loss'])
            for key in epoch_loss.keys():
                self.tb_writer.add_scalar('train/'+key, np.mean(epoch_loss[key]), global_step)
            
            self.validate(epoch, global_step)
            
            logging.info('--------------saving latest ckpt----------------')
            model_name = os.path.join(self.args.save_dir, 'latest.pkl')
            if os.path.exists(model_name):
                os.remove(model_name)
            t.save({'mainstream': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'model_D': self.model_D.state_dict() if self.model_D is not None else None,
                    'optimizer_D': self.optimizer_D.state_dict() if self.optimizer_D is not None else None,
                    'lr_scheduler_D': self.lr_scheduler_D.state_dict() if self.lr_scheduler_D is not None else None,
                    'epoch':epoch},
                    model_name)
            
            # if epoch == 14:
            #     model_name = os.path.join(self.args.save_dir, 'ep14.pkl')
            #     t.save({'mainstream': self.model.state_dict(),
            #             'optimizer': self.optimizer.state_dict(),
            #             'lr_scheduler': self.lr_scheduler.state_dict(),
            #             'epoch':epoch},
            #             model_name)
            
            #for csl1 no dev split only
            if 'step' in self.args_lr_sch['name'] and epoch in [59,64,69,74,79]:
                logging.info('----------save epoch {:d} ckpt------------'.format(epoch))
                model_name = os.path.join(self.args.save_dir, 'ep{:d}.pkl'.format(epoch))
                t.save({'mainstream': self.model.state_dict(), 
                        'optimizer': self.optimizer.state_dict()}, 
                        model_name)
            
            # break out by learning rate
            if self.lr_scheduler.optimizer.param_groups[0]["lr"] < 0.1 * self.args_opt['lr']:
                break
        
        self.tb_writer.close()
        os.remove(os.path.join(self.args.save_dir, 'latest.pkl'))  #finish and delete
    
    
    def validate(self, epoch, global_step, split='dev', model_file=None):
        #********************Validation and Test******************
        if self.args.mode == 'test':
            self.decoded_dict = {}
            self.wsize_dict = {}
            # self.head_share_ori = dict(np.load('results/vgg11_aaai22_sub/lcasan_vgg11_rpe_gau_head_share_qk10_modefromoriQ/wsize_fail.npz'))
            assert model_file is not None
            logging.info('----------------------Test--------------------------')
            logging.info('Restoring full model parameters from {:s}'.format(model_file))
            state_dict = update_dict(t.load(model_file)['mainstream'])
            self.model.load_state_dict(state_dict, strict=True)
            self.model.cuda()
            # qual_res_lst = [163]
        
        dset = self.create_dataloader(split=split, bsize=1)
        val_err = np.zeros([4])
        val_correct, val_count, val_loss, val_D = 0, 0, 0.0, 0.0
        wer_lst, gate_lst, ratio_lst, att_lst_1, att_lst_2, dist_lst = [], [], [], [], [], []
        for i, batch_data in tqdm(enumerate(dset), desc='[{:s} phase, epoch {:d}]'.format(split.upper(), epoch)):
            err, correct, count, plot = self.eval_batch(batch_data, need_att=True)
            val_err += err
            val_correct += correct
            val_count += count
            val_loss += plot['loss']
            if plot['att_2'] is not None:
                val_D += plot['att_2'].mean().item()
            # if i in qual_res_lst:
            #     print(batch_data['len_label'])
            #     print(batch_data['label'])
            if self.args.mode == 'test':
                wer_lst.append(err[0])
                ratio_lst.append(plot['ratio'])
                att_lst_1.append(plot['att_1'])
                att_lst_2.append(plot['att_2'])
                dist_lst.append(plot['miscell'])
        
        if self.args.mode == 'test':
            if self.args.mod_D is not None:
                if len(self.wsize_dict) != 0:
                    np.savez(self.args.save_dir+'/wsize.npz', **self.wsize_dict)
                np.savez(self.args.save_dir+'/D_wer.npz', D_1=t.cat(att_lst_1, dim=0).numpy(), D_2=t.cat(att_lst_2, dim=0).numpy(), wer=np.array(wer_lst))
            if self.args.sema_cons is not None and dist_lst[0] is not None:
                np.savez(self.args.save_dir+'/sc_dist.npz', dist=np.stack(dist_lst, axis=0))
        
        logging.info('-' * 50)
        logging.info('{:s} ACC: {:.5f}, {:d}/{:d}'.format(split.upper(), val_correct / val_count, val_correct, val_count))
        logging.info('{:s} WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(\
            split.upper(), val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))
        logging.info('{:s} LOSS: {:.5f}'.format(split.upper(), val_loss / val_count))

        if '2014' in self.args_data['dataset']:
            # ******Evaluation with official script (merge synonyms)***
            list_str_for_test = []
            for k, v in self.decoded_dict.items():
                start_time = 0
                for wi in v:
                    tl = np.random.random() * 0.1
                    list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                              list(dset.dataset.vocab.table.keys())[wi]))
                    start_time += tl
            
            tmp_prefix = str(uuid.uuid1())
            txt_file = '{:s}.txt'.format(tmp_prefix)
            result_file = os.path.join('evaluation_relaxation', txt_file)
            with open(result_file, 'w') as fid:
                fid.writelines(list_str_for_test)
            phoenix_eval_err = get_phoenix_wer(txt_file, split, tmp_prefix, dataset=self.args_data['dataset'])
            logging.info('[Relaxation Evaluation] {:s} WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(\
                split.upper(), phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
        else:
            phoenix_eval_err = list(val_err/val_count)
        
        if self.args_data['dataset'] == 'tvb' and self.args.mode == 'test':
            import pandas as pd
            df = pd.DataFrame(list(self.decoded_dict.items()), columns=['id', 'hypothesis'])
            df['hypothesis'] = df['hypothesis'].apply(lambda x: ''.join(self.vocab[i] for i in x))
            fname = self.args.save_dir + '/' + str(split) + '_hyp.csv'
            df.to_csv(fname, index=False, sep=',')
            
        if self.args.mode == 'train':
            self.tb_writer.add_scalar('valid/valid_wer', phoenix_eval_err[0], global_step)
            self.tb_writer.add_scalars('valid/valid_wer_scores', 
                                      {'SUB': phoenix_eval_err[1], 
                                        'INS': phoenix_eval_err[2], 
                                        'DEL': phoenix_eval_err[3]},
                                      global_step)
            self.tb_writer.add_scalar('valid/valid_loss', val_loss / val_count, global_step)
            self.tb_writer.add_scalar('valid/valid_D', val_D / val_count, global_step)
            
            #**********************Save checkpoints***************************
            model_name = os.path.join(self.args.save_dir, 'ep{:d}_step{:d}_wer{:.5f}_sub{:.2f}_ins{:.2f}_del{:.2f}.pkl'.format(\
                epoch, global_step, phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
            t.save({'mainstream': self.model.state_dict()}, model_name)
            self.model_manager.update(model_name, phoenix_eval_err, epoch, self.lr_scheduler, self.lr_scheduler_D, self.args_lr_sch['name'])
        
        del dset
        # with open(self.args.save_dir+'/predictions.pkl', 'wb') as f:
        #     pickle.dump(self.decoded_dict, f)

    def vis_fde(self, model_file):
        import cv2
        state_dict = update_dict(t.load(model_file)['mainstream'])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.cuda()
        self.model.eval()  # set NOT self.train in fde.py
        split = 'test'
        
        # ATTENTION: random drop issue
        if split == 'train':
            idx_lst = [80, 335, 362, 381, 3474, 3508, 3586, 631]
            self.args_data['p_drop'] = 0.9
        else:
            idx_lst = [0,20,50,60,70,71,72,73,80,90]
            # idx_lst = [80,90]
            self.args_data['p_drop'] = 0.5
        # idx_lst = [335]
        dset = self.create_dataloader(split=split, bsize=1)
        save_sg = []
        for i, batch_data in tqdm(enumerate(dset), desc='[VIS_FDE phase]'):
            if i > max(idx_lst):
                break
            elif i not in idx_lst:
                continue
            else:
                t.cuda.empty_cache()
                video = t.cat(batch_data['video']).cuda()
                len_of_video = batch_data['len_video'].cuda()
                signer = batch_data['signer']
                for k in range(len(signer)):
                    signer[k] = t.tensor(signer[k]).expand(len_of_video[k])
                signer = t.cat(signer, dim=0).cuda()
                if self.args.pose in ['prior', 'super_att'] and split == 'train':
                    heatmap = []
                    for hmap in zip(*batch_data['heatmap']):
                        heatmap.append(t.cat(hmap).cuda())
                    heatmap = heatmap[0]

                op_dict = self.model(video, len_of_video, signer=signer)
                sg, sg2, cam = op_dict['cam']  #[T,1,H,W]
                save_sg.append(sg.detach().cpu())
                continue
                cg, ch_cam = op_dict['ch_cam']  #[T,C,1,1]
                mask_lst = op_dict['spat_att']
                # cg = cg.squeeze()
                # if cam is None:
                #     cam = t.zeros_like(sg)
                # if ch_cam is None:
                #     ch_cam = t.zeros_like(cg)
                # ch_cam = ch_cam.squeeze()
                
                mean = t.tensor([0.5405, 0.5306, 0.5235]).reshape(1,3,1,1).cuda()
                video += mean
                video = np.uint8(255*video.cpu().numpy()).transpose(2,3,1,0)
                T = len_of_video.item()
                for j in tqdm(range(T)):
                    spat = sg[j, 0, ...].detach().cpu().numpy()  #[H,W]
                    if cam is not None:
                        spat_cam = cam[j, 0, ...].detach().cpu().numpy()  #[H,W]
                    if sg2 is not None:
                        spat2 = sg2[j, 0, ...].detach().cpu().numpy()
                    # chan = cg[j, ...].detach().cpu().numpy()
                    # chan_cam = ch_cam[j, ...].detach().cpu().numpy()
                    img = video[..., j]
                    # plt.figure(i+j)

                    # spatial attention mask
                    # spat = (spat - spat.min()) / (spat.max() - spat.min() + 1e-8)
                    # plt.subplot(1,5,1)
                    s = cv2.applyColorMap(np.uint8(255*spat), cv2.COLORMAP_JET)
                    s = cv2.resize(s, (224,224))
                    comb = np.uint8(0.5*img[..., ::-1] + 0.5*s)
                    # plt.axis('off')
                    # plt.imshow(comb)
                    cv2.imwrite(self.args.save_dir+'/vis_fde/'+str(split)+'_'+str(i)+'_'+str(j)+'_comb.jpg', comb)

                    if split == 'train':
                        hmap = heatmap[j, 0, ...].cpu().numpy()  #[H,W]

                        # dual_spat
                        # plt.subplot(1,5,2)
                        # s = cv2.applyColorMap(np.uint8(255*spat2), cv2.COLORMAP_JET)
                        # s = cv2.resize(s, (224,224))[..., ::-1]
                        # comb = np.uint8(0.5*img + 0.5*s)
                        # plt.axis('off')
                        # plt.imshow(comb)

                        # plt.subplot(1,5,3)
                        # s = cv2.applyColorMap(np.uint8(255*spat*spat2), cv2.COLORMAP_JET)
                        # s = cv2.resize(s, (224,224))[..., ::-1]
                        # comb = np.uint8(0.5*img + 0.5*s)
                        # plt.axis('off')
                        # plt.imshow(comb)

                        # cam
                        # plt.subplot(1,5,2)
                        # s = cv2.applyColorMap(np.uint8(255*spat_cam), cv2.COLORMAP_JET)
                        # s = cv2.resize(s, (224,224))[..., ::-1]
                        # comb = np.uint8(0.5*img + 0.5*s)
                        # plt.axis('off')
                        # plt.imshow(comb)

                        # reversed cam
                        # plt.subplot(1,5,3)
                        # s = cv2.applyColorMap(np.uint8(255*(1.0-spat_cam)), cv2.COLORMAP_JET)
                        # s = cv2.resize(s, (224,224))[..., ::-1]
                        # comb = np.uint8(0.5*img + 0.5*s)
                        # plt.axis('off')
                        # plt.imshow(comb)
                        
                        # pose heatmap
                        # plt.subplot(1,5,2)
                        # s = cv2.applyColorMap(np.uint8(255*hmap), cv2.COLORMAP_JET)
                        # s = cv2.resize(s, (224,224))[..., ::-1]
                        # comb = np.uint8(0.5*img + 0.5*s)
                        # plt.axis('off')
                        # plt.imshow(comb)

                        # pose modulated reversed cam
                        # plt.subplot(1,5,5)
                        # s = cv2.applyColorMap(np.uint8(255*(1.0-spat_cam)*hmap), cv2.COLORMAP_JET)
                        # s = cv2.resize(s, (224,224))[..., ::-1]
                        # comb = np.uint8(0.5*img + 0.5*s)
                        # plt.axis('off')
                        # plt.imshow(comb)
                    
                    # plt.savefig(self.args.save_dir+'/vis_fde/{:s}_{:d}_{:d}.jpg'.format(split, i, j))

                    # plt.figure(i+j+1000)
                    # x = np.arange(chan.shape[0])
                    # plt.plot(x, 1.0-chan_cam, color='orange', label='reversed channel cam')
                    # plt.plot(x, chan, color='b', label='channel gates')
                    # plt.xlabel("channel index")
                    # plt.ylabel("magnitude")
                    # plt.legend()
                    # plt.savefig(self.args.save_dir+'/vis_fde/{:s}_channel_{:d}_{:d}.jpg'.format(split, i, j))
        save_sg = t.cat(save_sg, dim=0).numpy()
        np.savez_compressed(self.args.save_dir+'/sg.npz', save_sg)


    def vis_vit(self, model_file):
        import cv2
        state_dict = update_dict(t.load(model_file)['mainstream'])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.cuda()
        self.model.eval()
        dset = self.create_dataloader(split='dev', bsize=1)
        # ATTENTION: random drop issue
        for i, batch_data in tqdm(enumerate(dset), desc='[VIS_VIT phase]'):
            if i==0:
                video = t.cat(batch_data['video']).cuda()
                len_of_video = batch_data['len_video'].cuda()
                coord = None
                if self.args.pose is not None:
                    coord = t.cat(batch_data['coord']).cuda()
                
                op_dict = self.model(video, len_of_video, coord)
                attn_scores = t.stack(op_dict['spat_att'], dim=1).mean(dim=2)  # [T,L,N], N=(224//16)*(224//16)+2=14*14+2=196+2
                mean = t.tensor([0.5372,0.5273,0.5195]).reshape(1,3,1,1).cuda()
                video += mean
                video = np.uint8(255*video.cpu().numpy()).transpose(2,3,1,0)
                T = len_of_video.item()
                L = attn_scores.shape[1]
                print('1st:', attn_scores[0,-2,:2], '2nd:', attn_scores[0,-1,:2])
                # for j in tqdm(range(T)):
                #     score = attn_scores[j,:,2:].unsqueeze(-1)  #[L,196,1]
                #     assert score.shape[1] == 196
                #     score = score.expand(-1,-1,256).reshape(L,14,14,16,16).permute(0,1,3,2,4).reshape(L,224,224)
                #     score = score.detach().cpu().numpy()
                #     img = video[..., j]
                #     plt.figure(j)
                #     for k in range(L):
                #         s = score[k, ...]
                #         if k==0:
                #             print(s.max(), s.min(), s.sum()/256, np.std(s))
                #         s = cv2.applyColorMap(np.uint8(255*s), cv2.COLORMAP_JET)[..., ::-1]
                #         comb = np.uint8(0.5*img + 0.5*s)
                #         plt.subplot(2, L//2, k+1)
                #         plt.imshow(comb)
                #         plt.axis('off')
                #     plt.savefig(self.args.save_dir+'/vis_vit/0_head0_{:d}.jpg'.format(j))
            else:
                break


    def vis_attention(self, model_file):
        def _lin_normalize(hmaps):
            min = hmaps.min(axis=(-2,-1), keepdims=True)
            max = hmaps.max(axis=(-2,-1), keepdims=True)
            hmaps = (hmaps - min) / (max - min + 1e-6)
            return hmaps
        
        import cv2
        from torchvision.transforms.functional import resize
        state_dict = update_dict(t.load(model_file)['mainstream'])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.cuda()
        self.model.eval()
        
        split = 'test'
        if split == 'train':
            idx_lst = [80, 335, 362, 381, 3474, 3508, 3586, 631]
        else:
            idx_lst = [0,20,50,60,70,71,72,73,80,90]
            # idx_lst = [80,90]
        self.args_data['p_drop'] = 0.5
        dset = self.create_dataloader(split=split, bsize=1)
        # idx = 5  # ATTENTION: random drop issue
        save_sg = []  #spatial gates
        for i, batch_data in tqdm(enumerate(dset), desc='[VIS_ATT phase]'):
            if i > max(idx_lst):
                break
            elif i not in idx_lst:
                continue
            else:
                t.cuda.empty_cache()
                video = t.cat(batch_data['video']).cuda()
                len_of_video = batch_data['len_video'].cuda()
                heatmap = []
                for hmap in zip(*batch_data['heatmap']):
                    heatmap.append(t.cat(hmap).cuda()) 
                
                op_dict = self.model(video, len_of_video)
                offset_lst, mask_lst = op_dict['offset'], op_dict['spat_att']

                # cw = offset_lst[0].squeeze().detach().cpu().numpy()
                # print(cw.shape)
                # np.savez(self.args.save_dir+'/cw.npz', cw=cw)
                # break

                video = video.cpu()
                if self.args_data['dataset'] == '2014':
                    mean = t.tensor([0.5372, 0.5273, 0.5195]).reshape(1,3,1,1)
                elif self.args_data['dataset'] == '2014SI':
                    mean = t.tensor([0.5405, 0.5306, 0.5235]).reshape(1,3,1,1)
                video += mean
                # video = transforms.functional.resize(video, hmaps.shape[-2:])
                video = np.uint8(255*video.detach().cpu().numpy()).transpose(2,3,1,0)
                
                # visualize mask
                if self.args_model['vis_mod'] == 'mb_v2':
                    mask_lst = mask_lst[1:]
                elif self.args_model['vis_mod'] == 'resnet18':
                    mask_lst = mask_lst[2:]
                else:
                    masks = mask_lst[0]

                save_sg.append(masks.detach().cpu());continue
                assert self.args.spatial_att == 'cbam'
                for idx in tqdm(range(0, masks.shape[0])):
                    mask = masks[idx, ...]
                    mask = mask.squeeze().detach().cpu().numpy()
                    # mask_label = heatmap[0][idx, 0, ...].detach().cpu().numpy()
                    # mask = _lin_normalize(mask)
                    # print(mask.max(), mask_label.max(), mask.min(), mask_label.min())

                    mask = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
                    img = video[..., idx][..., ::-1]
                    comb = np.uint8(0.5*img + 0.5*cv2.resize(mask, (224,224)))
                    cv2.imwrite(self.args.save_dir+'/vis_fde/'+str(split)+'_'+str(i)+'_'+str(idx)+'_comb.jpg', comb)
                    # cv2.imwrite(self.args.save_dir+'/vis/'+str(i)+'_'+str(idx)+'_mask.jpg', mask)
                
                    if self.args.cbam_pool in ['softmax', 'max_softmax']:
                        # distribution of channel weight
                        c_w = offset_lst[0]
                        c_w = c_w.squeeze().detach().cpu().numpy()[idx, :]  #[512]
                        x = np.arange(0, c_w.shape[0])
                        # plt.figure(4*idx+1)
                        # ax = plt.subplot(111)
                        # ax.spines['right'].set_visible(False)
                        # ax.spines['top'].set_visible(False)
                        # ax.set_xlabel('channel index')
                        # ax.set_ylabel('weight')
                        # ax.scatter(x, c_w, s=10)
                        # y = (1.0/512)*np.ones(x.shape[0])
                        # ax.plot(x, y, color='r', ls='--', label='1/512')
                        # ax.legend()
                        # plt.savefig(self.args.save_dir+'/vis/'+str(i)+'_'+str(idx)+'_cw.jpg', bbox_inches='tight', pad_inches=0)
        save_sg = t.cat(save_sg, dim=0).numpy()
        np.savez_compressed(self.args.save_dir+'/sg.npz', save_sg)

# def vis_cam(self, model_file):
    #     import cv2
    #     from vis.gradcam import CAM
    #     from vis.image import show_cam_on_image
    #     state_dict = update_dict(t.load(model_file)['mainstream'])
    #     self.model.load_state_dict(state_dict, strict=True)
    #     target_layer = self.model.vis_mod.CNN_stack[-4]
    #     cam = CAM(model=self.model, target_layer=target_layer, use_cuda=True)
        
    #     dset = self.create_dataloader(split='dev', bsize=1)
    #     for i, batch_data in tqdm(enumerate(dset), desc='[VIS_CAM phase]'):
    #         video = t.cat(batch_data['video']).cuda()
    #         len_video = batch_data['len_video'].cuda()
    #         label = batch_data['label']
    #         grayscale_cam = cam(video=video, len_video=len_video, 
    #                             method='gradcam', target_category=label)
    #         rgb_img = video.cpu().data.numpy().transpose(0,2,3,1)  #[T,H,W,3]
    #         visualization = show_cam_on_image(rgb_img, grayscale_cam)
    #         for j in range(video.shape[0]):
    #             cv2.imwrite(self.args.save_dir+'/cam/'+str(j)+'.jpg', visualization[j])
    #         if i==0:
    #             break

# def get_hyp_after_lm(self, pred_seq, beam_scores, out_seq_len):
    #     '''
    #     Parameters
    #     ----------
    #     pred_seq : tensor [N_BEAMS,N_TIMESTEPS]
    #         DESCRIPTION.
    #     beam_scores : tensor [N_BEAMS]
    #         -log(p). 
    #     out_seq_len : tensor [N_BEAMS]
    #         Values after out_seq_len are non-sensical.

    #     Returns
    #     -------
    #     hyp after lm
    #     '''
    #     assert self.lm is not None
    #     n_beams = beam_scores.shape[0]
    #     assert n_beams == self.args.rec_beam_size
    #     for j in range(n_beams):
    #         hyp = [x[0] for x in groupby(pred_seq[j][:out_seq_len[j]].tolist())]
    #         sen = [self.vocab[x] for x in hyp]
    #         if sen == []:
    #             continue
    #         beam_scores[j] -= self.args.lm_weight * self.lm.log_s(sen)
        
    #     best_beam = t.argmin(beam_scores)
    #     return [x[0] for x in groupby(pred_seq[best_beam][:out_seq_len[best_beam]].tolist())]