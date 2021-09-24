# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:33:01 2020

@author: Ronglai ZUO
Script for base class
"""
import torch as t; t.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import logging
import uuid
import arpa

from model import SLRModel, CMA
from modules.dcn import gen_mask_from_pose
from phoenix_datasets import PhoenixVideoTextDataset, PhoenixTVideoTextDataset, PhoenixSIVideoTextDataset, CSLVideoTextDataset, CSLDailyVideoTextDataset
from utils.metric import get_wer_delsubins
from utils.utils import update_dict, worker_init_fn, LossManager, ModelManager, freeze_params, unfreeze_params, record_loss
from utils.figure import gen_att_map
from evaluation_relaxation.phoenix_eval import get_phoenix_wer
from ctcdecode import CTCBeamDecoder
from modules.searcher import CTC_decoder

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
        if self.args_model['name'] == 'lcsa' or self.args_model['name'] == 'fcn':
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
                                  sema_cons=args.sema_cons
                                  )
        elif self.args_model['name'] == 'CMA':
            self.model = CMA(gls_voc_size=self.voc_size)
        
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()
        self.criterion = nn.CTCLoss(self.blank_id, zero_infinity=True).cuda()
        if args.ve:
            self.ve_crit = nn.CTCLoss(self.blank_id, zero_infinity=True).cuda()
        # if args.va:
        #     self.va_crit = SeqKD(T=8, blank_id=self.blank_id).cuda()

        if args.pose is not None and ('deform' in args.pose or args.pose=='super_att'):
            # self.pose_crit = nn.SmoothL1Loss().cuda()
            self.pose_crit = nn.MSELoss().cuda()
        
        ctc_decoder_vocab = [chr(x) for x in range(20000, 20000 + self.voc_size)]
        self.ctc_decoder = CTCBeamDecoder(ctc_decoder_vocab,
                                          beam_width=args.beam_size,
                                          blank_id=self.blank_id,
                                          num_processes=5
                                          )

        self.decoded_dict = {}
        self.dset_dict = {'2014': {'cls': PhoenixVideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-multisigner', 'mean': [0.5372,0.5273,0.5195], 'hmap_mean': [0.0236, 0.0250, 0.0164, 0.0283, 0.0305, 0.0240, 0.0564]},
                          '2014T': {'cls': PhoenixTVideoTextDataset, 'root': '../../data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T', 'mean': [0.5372,0.5273,0.5195], 'hmap_mean': []},
                          '2014SI': {'cls': PhoenixSIVideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-signerindependent-SI5', 'mean': [0.5405, 0.5306, 0.5235], 'hmap_mean': []},
                          'csl1': {'cls': CSLVideoTextDataset, 'root': ('../../data/ustc-csl', 'split_1.txt'), 'mean': [0.5827, 0.5742, 0.5768], 'hmap_mean': []},
                          'csl2': {'cls': CSLVideoTextDataset, 'root': ('../../data/ustc-csl', 'split_2.txt'), 'mean': [0.5827, 0.5742, 0.5768], 'hmap_mean': []},
                          'csl-daily': {'cls': CSLDailyVideoTextDataset, 'root': '../../data/csl-daily', 'mean': [0.6868, 0.6655, 0.6375], 'hmap_mean': []}}
        if args.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir=args.save_dir + "/tensorboard/")
    
    
    def _create_optimizer(self):
        #filter for DCN
        if self.args_opt['name'] == 'adam':
            return t.optim.Adam([{'params': [p for n, p in self.model.named_parameters() if 'conv_offset' not in n]},
                                 {'params': [p for n, p in self.model.named_parameters() if 'conv_offset' in n], 'lr': self.args_opt['lr']*self.args_dcn['lr_factor']},
                                 ],
                                self.args_opt['lr'], self.args_opt['betas'], self.args_opt['weight_decay'])
        elif self.args_opt['name'] == 'sgd':
            return t.optim.SGD(self.model.parameters(), self.args_opt['lr'], self.args_opt['momentum'], self.args_opt['weight_decay'])
        else:
            raise KeyError('We only support adam and sgd now.\n')
            
            
    def _create_lr_scheduler(self):
        if self.args_lr_sch['name'] == 'plateau':
            return t.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                verbose=False,
                threshold_mode="abs",
                factor=self.args_lr_sch['decrease_factor'],
                patience=self.args_lr_sch['patience'],  #6 eval steps!
            )
        elif self.args_lr_sch['name'] == 'step':
            return t.optim.lr_scheduler.StepLR(optimizer=self.optimizer, 
                                               step_size=2,  #2 epochs!
                                               gamma=self.args_lr_sch['decrease_factor'])
        elif self.args_lr_sch['name'] == 'mstep':
            return t.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                    # milestones=[20,30,40,50,60,65,70,75],
                                                    milestones=[15,25,30,35,40,45],
                                                    gamma=self.args_lr_sch['decrease_factor'])
        else:
            raise KeyError('We only support plateau, StepLR, and MultiStepLR scheduler now.\n')
            
            
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
                use_random=use_random,
                pose=self.args.pose,
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
                                shuffle=True if split=='train' else False,
                                sampler=spler,
                                num_workers=8,
                                # worker_init_fn=worker_init_fn if self.args.batch_size==4 else None,
                                collate_fn=dataset.collate_fn,
                                drop_last=False)
        return dataloader
    
    
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
    
    
    def eval_batch(self, batch_data, need_att=False):
        with t.no_grad():
            batch_size = len(batch_data['video'])
            video = t.cat(batch_data['video']).cuda()
            len_video = batch_data['len_video'].cuda()
            label = batch_data['label'].cuda()
            len_label = batch_data['len_label'].cuda()
            video_id = batch_data['id']
            heatmap = [None]
            if self.args.pose in ['filter', 'modality']:
                heatmap = []
                for hmap in zip(*batch_data['heatmap']):
                    heatmap.append(t.cat(hmap).cuda()) 
            
            self.model.eval()
            op_dict = self.model(video, len_video, heatmap[0], return_att=need_att)
            gls_logits, len_video, plot_lst = op_dict['gls_logits'], op_dict['len_video'], op_dict['plot']
            
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
                hyp_lst.append(hyp)
                # hyp = self.get_hyp_after_lm(pred_seq[i], beam_scores[i], out_seq_len[i])
                
                id = ''.join(video_id[i])
                self.decoded_dict[id] = hyp
                
                correct += int(ref == hyp)
                colors = ['blue', 'brown', 'red', 'green', 'gold', 'pink', 'cyan', 'lime', 'purple', 'tomato', 'yellowgreen', 'maroon', 'black']
                # if correct and label.shape[0] <= len(colors) and id in self.head_share_ori.keys():
                #     # self.wsize_dict[id] = plot_lst[1][0,0,:,0].cpu().numpy()
                #     # self.wsize_dict[id] = gls_scores[0, :, label].cpu().numpy()
                #     D = plot_lst[1][0, 0, :, 0].cpu().numpy()
                #     D_ori = self.head_share_ori[id]
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
                #         y = gls_scores[0, :, l].cpu().numpy()
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
                
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1
                start = end
                #save attention heatmap
                # if need_att and i == 0 and self.args_model['name']=='lcsa':
                #     assert plot is not None
                #     logging.info('-------------generate attention heatmap--------------')
                #     id = ''.join(video_id[0])
                #     print(id, hyp, err_delsubins[0])
                #     gen_att_map(plot['att'], self.args.save_dir+'/'+str(err[0])+'_'+id+'.png')
                #     if self.args.comb_conv == 'gate':
                #         logging.info('mean of gate: {:.3f}'.format(t.mean(att['gate'][0,:len_video.cpu()[0],:])))
                #         gen_att_map(att['gate'][0,:len_video.cpu()[0],:], self.args.save_dir+'/gate'+str(err[0])+'_'+id+'.png')
                
            assert end == label.size(0)
        
        return err_delsubins, correct, count, {'loss': val_loss.item(), 
                                               # 'gate': t.mean(plot['gate'][0,:len_video.cpu()[0],:]).cpu().numpy() if self.args.comb_conv == 'gate' else None,
                                               'att_1': plot_lst[0].mean(dim=(-2,-1)).cpu() if need_att and self.args_model['seq_mod']=='transformer' and self.args_tf['pe']=='rpe_gau' else None,
                                               'att_2': plot_lst[1].mean(dim=(-2,-1)).cpu() if need_att and self.args_model['seq_mod']=='transformer' and self.args_tf['pe']=='rpe_gau' else None,
                                               'ratio': (len_video/len_label).cpu(),
                                               'hyp_lst': hyp_lst}
    
    
    def train_batch(self, batch_data, epoch):
        video = t.cat(batch_data['video']).cuda()
        len_of_video = batch_data['len_video'].cuda()
        label = batch_data['label'].cuda()
        len_label = batch_data['len_label'].cuda()
        heatmap = [None]
        if self.args.pose in ['deform', 'deform_all', 'deform_patch']:
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
        op_dict = self.model(video, len_of_video, heatmap[0])
        gls_logits, vis_logits, len_video, offset_lst, mask_lst = op_dict['gls_logits'], op_dict['vis_logits'], op_dict['len_video'], op_dict['dcn_offset'], op_dict['spat_att']
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
            
        if self.args.pose == 'super_att':
            if self.args_model['vis_mod'] == 'mb_v2':
                idx = 1
            elif self.args_model['vis_mod'] == 'vgg11':
                idx = 0
            elif self.args_model['vis_mod'] == 'resnet18':
                idx = 2
            
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
            
        loss += self.args.pose_f * loss_pose
        
        if self.args.ve:
            ve_vis_prob = vis_logits.log_softmax(-1).permute(1,0,2)
            loss_ve = self.ve_crit(ve_vis_prob, label, len_of_video, len_label)  #ctc loss
            loss += loss_ve
        # if self.args.va:
        #     alpha = 25.0
        #     loss_va = self.dc_crit(gls_scores, vis_fea.detach())  #visual to sequential, vis is label
        #     loss += alpha*loss_va
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {'loss': loss.item(), 'loss_pose': loss_pose.item()}
    
    
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
            start_epoch = saved_dict['epoch']+1
            global_step = self.len_dtrain // self.args_model['batch_size'] * start_epoch
            t.manual_seed(self.args.seed+start_epoch)  #change dataloader order
            dtrain = self.create_dataloader(split='train', bsize=self.args_model['batch_size'])
        
        for epoch in range(start_epoch, max_num_epoch):
            #*********************Training*******************
            epoch_loss = defaultdict(list)
            
            # if epoch == 15 and self.args.gfe == 1:
            #     ms_dir = os.path.join(self.args.save_dir, 'mainstream_ckpt')
            #     if not os.path.exists(ms_dir):
            #         os.makedirs(ms_dir)
            #     model_name = os.path.join(ms_dir, 'ep{:d}.pkl'.format(self.args.start_gfe_epoch-1))
            #     logging.info('-----------saving mainstream ckpt to {:s}-------'.format(model_name))
            #     t.save({'mainstream': self.model.state_dict(), 
            #             'optimizer': self.optimizer.state_dict()}, 
            #             model_name)
            
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
                    'epoch':epoch},
                    model_name)
            
            #for csl1 no dev split only
            if 'step' in self.args_lr_sch['name'] and epoch in [59,64,69,74,79]:
                logging.info('----------save epoch {:d} ckpt------------'.format(epoch))
                model_name = os.path.join(self.args.save_dir, 'ep{:d}.pkl'.format(epoch))
                t.save({'mainstream': self.model.state_dict(), 
                        'optimizer': self.optimizer.state_dict()}, 
                        model_name)
            
            # break out by learning rate
            if self.lr_scheduler.optimizer.param_groups[0]["lr"] < 1e-5:
                break
        
        self.tb_writer.close()
        os.remove(os.path.join(self.args.save_dir, 'latest.pkl'))  #finish and delete
    
    
    def validate(self, epoch, global_step, split='dev', model_file=None):
        #********************Validation and Test******************
        if self.args.mode == 'test':
            self.decoded_dict = {}
            self.wsize_dict = {}
            # self.head_share_ori = dict(np.load('results/lcasan_vgg11_rpe_gau_head_share_qk10_modefromoriQ/wsize_fail.npz'))
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
        wer_lst, gate_lst, ratio_lst, att_lst_1, att_lst_2 = [], [], [], [], []
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
                # if self.args.comb_conv == 'gate':
                #     gate_lst.append(plot['gate'])
                wer_lst.append(err[0])
                ratio_lst.append(plot['ratio'])
                att_lst_1.append(plot['att_1'])
                att_lst_2.append(plot['att_2'])
                # if err[0] > 0.5:
                #     print(batch_data['len_label'])
        
        if self.args.mode == 'test':
            # if self.args.comb_conv == 'gate':
            #     np.savez(self.args.save_dir+'/gate_wer.npz', gate=np.array(gate_lst), wer=np.array(wer_lst))
            # np.savez(self.args.save_dir+'/wer_ratio_'+self.args.test_split+'.npz', wer=np.array(wer_lst), ratio=np.array(ratio_lst))
            if self.args.mod_D is not None:
                if len(self.wsize_dict) != 0:
                    np.savez(self.args.save_dir+'/wsize.npz', **self.wsize_dict)
                np.savez(self.args.save_dir+'/D_wer.npz', D_1=t.cat(att_lst_1, dim=0).numpy(), D_2=t.cat(att_lst_2, dim=0).numpy(), wer=np.array(wer_lst))
        
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
            self.model_manager.update(model_name, phoenix_eval_err, epoch, self.lr_scheduler, self.args_lr_sch['name'])
        
        del dset
    
    
    def vis_cam(self, model_file):
        import cv2
        from vis.gradcam import CAM
        from vis.image import show_cam_on_image
        state_dict = update_dict(t.load(model_file)['mainstream'])
        self.model.load_state_dict(state_dict, strict=True)
        target_layer = self.model.vis_mod.CNN_stack[-4]
        cam = CAM(model=self.model, target_layer=target_layer, use_cuda=True)
        
        dset = self.create_dataloader(split='dev', bsize=1)
        for i, batch_data in tqdm(enumerate(dset), desc='[VIS_CAM phase]'):
            video = t.cat(batch_data['video']).cuda()
            len_video = batch_data['len_video'].cuda()
            label = batch_data['label']
            grayscale_cam = cam(video=video, len_video=len_video, 
                                method='gradcam', target_category=label)
            rgb_img = video.cpu().data.numpy().transpose(0,2,3,1)  #[T,H,W,3]
            visualization = show_cam_on_image(rgb_img, grayscale_cam)
            for j in range(video.shape[0]):
                cv2.imwrite(self.args.save_dir+'/cam/'+str(j)+'.jpg', visualization[j])
            if i==0:
                break


    def vis_attention(self, model_file):
        import cv2
        from torchvision.transforms.functional import resize
        state_dict = update_dict(t.load(model_file)['mainstream'])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.cuda()
        self.model.eval()
        
        dset = self.create_dataloader(split='dev', bsize=1)
        idx = 5  # ATTENTION: random drop issue
        for i, batch_data in tqdm(enumerate(dset), desc='[VIS_ATT phase]'):
            video = t.cat(batch_data['video']).cuda()
            len_of_video = batch_data['len_video'].cuda()
            coord = t.cat(batch_data['coord'])
            coord = coord[idx, ...]  #[3,2]
            heatmap = []
            for hmap in zip(*batch_data['heatmap']):
                heatmap.append(t.cat(hmap).cuda()) 
            
            op_dict = self.model(video, len_of_video, None)
            offset_lst, mask_lst = op_dict['dcn_offset'], op_dict['spat_att']
            
            if self.args.spatial_att in ['dcn', 'dcn_nooff']:
                offset_lst = offset_lst[:2]  #first two layer offsets
                H = offset_lst[1].shape[-1]
                corners = [t.LongTensor([H//8,H//8]), t.LongTensor([H-H//8,H//8]), t.LongTensor([H//8,H-H//8]), t.LongTensor([H-H//8,H-H//8])]
                mask_labels = gen_mask_from_pose(offset_lst[0], heatmap[0]).detach()
                
                # visualize offsets
                for j in range(2):
                    offset_lst[j] = offset_lst[j][idx, ...]  #[18,28,28]
                for j in range(9):
                    offset = offset_lst[0]
                    if j<3:
                        center = ((H-1) * coord[j, ...]).long()
                    elif j==3:
                        center = t.LongTensor([H//2,H//2])
                    elif j==4:
                        center = t.LongTensor([H-3,H//2])
                    else:
                        center = corners[j-5]
                    x, y = offset.split(9, dim=0)  #[9,28,28]
                    img = video[idx, ...]  #[3,224,224]
                    img = img.permute(1,2,0).cpu() + t.Tensor([0.5372,0.5273,0.5195])
                    img = resize(img.permute(2,0,1), [H,H])
                    img = img.permute(1,2,0).numpy().copy()
                    img = (255*img).astype(np.uint8)[..., ::-1]
                    img = cv2.UMat(img).get()
                
                    first = []
                    off_lst = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
                    for k in range(9):
                        tmp = center + t.Tensor(off_lst[k]) + t.Tensor((x[k,center[0],center[1]], y[k,center[0],center[1]]))
                        first.append(tmp.long())
                        if k!=4:
                            cv2.circle(img, (int(tmp[1]), int(tmp[0])), radius=2, color=(0,0,255))
                        else:
                            cv2.circle(img, (int(tmp[1]), int(tmp[0])), radius=2, color=(255,0,0))
                    
                    # offset = offset_lst[0]
                    # x, y = offset.split(9, dim=0)
                    # for cen in first:
                    #     for k in range(9):
                    #         tmp = cen + t.Tensor(off_lst[k]) + t.Tensor((x[k,cen[0],cen[1]], y[k,cen[0],cen[1]]))
                    #         tmp = tmp.clamp(0, H-1)
                    #         cv2.circle(img, (int(tmp[1]), int(tmp[0])), radius=1, color=(0,0,255))
                    
                    print(center)
                    cv2.circle(img, (center[1],center[0]), radius=1, color=(0,255,0))
                    path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_vis_deform_{:d}_{:d}.jpg'.format(H,j))
                    cv2.imwrite(path, img)
                    
                # visualize center kernel position
                center_offset_x, center_offset_y = offset[4, ...], offset[13, ...]
                c = t.stack(t.meshgrid(t.arange(H), t.arange(H)), dim=0).view(2,-1)
                img = video[idx, ...]  #[3,224,224]
                img = img.permute(1,2,0).cpu() + t.Tensor([0.5372,0.5273,0.5195])
                img = resize(img.permute(2,0,1), [H,H])
                img = img.permute(1,2,0).numpy().copy()
                img = (255*img).astype(np.uint8)[..., ::-1]
                img = cv2.UMat(img).get()
                for j in range(H**2):
                    x, y = c[:, j]
                    o_x, o_y = center_offset_x[x, y], center_offset_y[x, y]
                    cv2.circle(img, (int(x+o_x), int(y+o_y)), radius=1, color=(255,0,0))
                path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_vis_deform_center_offsets.jpg')
                cv2.imwrite(path, img)
            
            # visualize mask
            if self.args_model['vis_mod'] == 'mb_v2':
                mask_lst = mask_lst[1:]
            elif self.args_model['vis_mod'] == 'resnet18':
                mask_lst = mask_lst[2:]
            else:
                mask_lst = mask_lst[0:]
            m_i = 0
            for masks in mask_lst:
                if masks is None:
                    continue
                m_i += 1
                if self.args.spatial_att in ['dcn', 'dcn_nooff']:
                    for j in range(9):
                        mask = masks[idx, j, ...]  #[28,28]
                        mask_label = mask_labels[idx, j, ...]
                        print(mask.max(), mask_label.max(), mask.min(), mask_label.min())
                        
                        mask = (255*mask).detach().cpu().numpy().astype(np.uint8)
                        # mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                        path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_mask_{:d}_{:d}.jpg'.format(m_i, j))
                        cv2.imwrite(path, mask)
                        
                        mask_label = (255*mask_label).cpu().numpy().astype(np.uint8)
                        # mask_label = cv2.applyColorMap(mask_label, cv2.COLORMAP_JET)
                        path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_mask_label_{:d}.jpg'.format(j))
                        cv2.imwrite(path, mask_label)
                else:
                    mask = masks[idx, ...]
                    if self.args.spatial_att == 'ca':
                        mask = mask.mean(dim=0)
                    mask = mask.squeeze()
                    mask_label = heatmap[0][idx, 0, ...]
                    print(mask.max(), mask_label.max(), mask.min(), mask_label.min())
                    
                    mask = (255*mask).detach().cpu().numpy().astype(np.uint8)
                    path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_mask_{:d}.jpg'.format(m_i))
                    cv2.imwrite(path, mask)
                    
                    mask_label = (255*mask_label).cpu().numpy().astype(np.uint8)
                    path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_mask_label.jpg')
                    cv2.imwrite(path, mask_label)
            
            if self.args.cbam_pool in ['softmax', 'max_softmax']:
                # distribution of channel weight
                count = 0
                for c_w in offset_lst:
                    if c_w is None:
                        continue
                    count += 1
                    c_w = c_w.squeeze().detach().cpu().numpy()[idx, :]  #[512]
                    x = np.arange(0, c_w.shape[0])
                    plt.figure(count)
                    plt.plot(x, c_w, label='weight of channels')
                    path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_weights_{:d}.jpg'.format(count))
                    plt.savefig(path)
            
            if i==0:
                break
