# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:29:25 2021

@author: Ronglai Zuo 
TrainingManager of semi-supervised learning
"""

import torch as t; t.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler

import numpy as np
import os
import logging

from modules.dcn import gen_mask_from_pose
from modules.criterion import SeqKD
from base import TrainingManager

from tqdm import tqdm
from collections import defaultdict
from itertools import groupby
from utils.utils import LossManager, ModelManager, record_loss
from utils.metric import get_wer_delsubins


class TrainingManagerSemi(TrainingManager):
    def __init__(self, args, vocab):
        super(TrainingManagerSemi, self).__init__(args, vocab)
        if args.va: 
            self.dc_crit = SeqKD(T=8, blank_id=self.blank_id).cuda()  # distribution consistency loss, T denotes temperature
        if args.sema_cons is not None:
            self.sema_crit = nn.TripletMarginWithDistanceLoss(distance_function=lambda x,y: 1.0-F.cosine_similarity(x,y), margin=2.0).cuda()
    
    def create_dataloader(self, split='train', bsize=2, use_random=None):
        if use_random is None:
            if split == 'train':
                use_random = True
            else:
                use_random = False

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
        
        if split == 'train':
            assert 'semi' in self.args.setting
            ratio = int(self.args.setting.split('_')[-1]) / 100
            fname = './phoenix_datasets/' + self.args.data['dataset'] + '_shuffle_idx.npz'
            if os.path.exists(fname):
                print('loading training set idx list from {:s}'.format(fname))
                train_idx = np.load(fname)['arr_0']
            else:
                train_idx = np.arange(len(dataset))
                np.random.shuffle(train_idx)
                np.savez(fname, train_idx)
            sampler_labeled = sampler.SubsetRandomSampler(train_idx[:int(len(dataset)*ratio)])
            sampler_unlabeled = sampler.SubsetRandomSampler(train_idx[int(len(dataset)*ratio):])
            self.num_labeled = int(len(dataset)*ratio)
            self.num_unlabeled = len(dataset) - self.num_labeled
            if 'ptf' in self.args.setting:
                # pretrain-then-finetune (ptf) use all data without label
                sampler_unlabeled = None
                self.num_unlabeled = len(dataset)
            
            dataloader_labeled = DataLoader(dataset, 
                                            bsize,
                                            shuffle=False,
                                            sampler=sampler_labeled,
                                            num_workers=8,
                                            # worker_init_fn=worker_init_fn if self.args.batch_size==4 else None,
                                            collate_fn=dataset.collate_fn,
                                            drop_last=False)
            dataloader_unlabeled = DataLoader(dataset, 
                                            bsize,
                                            shuffle=False,
                                            sampler=sampler_unlabeled,
                                            num_workers=8,
                                            # worker_init_fn=worker_init_fn if self.args.batch_size==4 else None,
                                            collate_fn=dataset.collate_fn,
                                            drop_last=False)
            return dataloader_labeled, dataloader_unlabeled
        
        else:
            return DataLoader(dataset, 
                              bsize,
                              shuffle=False,
                              sampler=None,
                              num_workers=8,
                              # worker_init_fn=worker_init_fn if self.args.batch_size==4 else None,
                              collate_fn=dataset.collate_fn,
                              drop_last=False)
    
    
    def train_batch(self, batch_data, epoch, phase='full', labeled=True, update=True):
        video = t.cat(batch_data['video']).cuda()
        len_of_video = batch_data['len_video'].cuda()
        if labeled:
            label = batch_data['label'].cuda()
            len_label = batch_data['len_label'].cuda()
        
        heatmap = [None]
        if self.args.pose == 'super_att':
            heatmap = []
            for hmap in zip(*batch_data['heatmap']):
                heatmap.append(t.cat(hmap).cuda())
        
        self.model.train()
        # gls_scores, len_video, spat_att, vis_fea, _ = self.model(video, len_of_video, heatmap[0])
        op_dict = self.model(video, len_of_video, heatmap[0])
        gls_logits, vis_logits, len_video, offset_lst, mask_lst, semantics = \
            op_dict['gls_logits'], op_dict['vis_logits'], op_dict['len_video'], op_dict['dcn_offset'], op_dict['spat_att'], op_dict['semantics']
        gls_prob = gls_logits.log_softmax(-1)
        gls_prob = gls_prob.permute(1,0,2)
        
        loss = t.tensor(0.0).cuda()
        loss_ctc_m = loss_ctc_v = loss_dc_v2s = loss_dc_s2v =  loss_sc = None
        # loss of ctc main and ctc visual
        if labeled:
            loss_ctc_m = self.criterion(gls_prob, label, len_video, len_label)  #ctc main
            loss += loss_ctc_m
            if self.args.ve:
                vis_prob = vis_logits.log_softmax(-1).permute(1,0,2)
                loss_ctc_v = self.criterion(vis_prob, label, len_of_video, len_label)  #ctc visual module
                loss += loss_ctc_v
        
        # pseudo labeling
        elif self.args.pl:
            gls_prob = F.softmax(gls_logits, dim=-1)
            pred_seq, _, _, out_seq_len = self.ctc_decoder.decode(gls_prob, len_video)
            p_label, p_len_label = [], []
            for i in range(len_video.shape[0]):
                hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                p_label.extend(hyp)
                p_len_label.append(len(hyp))
            p_label, p_len_label = t.tensor(p_label).cuda(), t.tensor(p_len_label).cuda()
            loss_ctc_m = self.criterion(gls_logits.log_softmax(-1).permute(1,0,2), p_label, len_video, p_len_label)
            # alpha_f = 3.0
            loss += loss_ctc_m #* alpha_f * (epoch - self.args.num_pretrain_epoch + 1) / (self.args.max_num_epoch - self.args.num_pretrain_epoch)
        
        # loss of distribution consistency which aims to train transformer
        if self.args.va:
            alpha = 25.0
            if labeled:
                loss_dc_s2v = self.dc_crit(vis_logits, gls_logits.detach())  #sequential to visual, seq is label
                loss += alpha * loss_dc_s2v
            else:
                loss_dc_v2s = self.dc_crit(gls_logits, vis_logits.detach())  #visual to sequential, vis is label
                loss += alpha * loss_dc_v2s
        
        # loss of semantic consistency
        if self.args.sema_cons is not None:
            anc, pos, neg = semantics
            loss_sc = self.sema_crit(anc, pos, neg)
            loss += loss_sc

        # MSE loss between spatial attention mask and pose heatmap
        loss_pose = t.tensor(0.0).cuda()
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
            elif self.args.spatial_att == 'cbam':
                for mask in masks:
                    if upsample:
                        mask = F.upsample(mask, size=self.args.heatmap_shape[0])
                    mask_label = heatmap[self.args.heatmap_shape.index(mask.shape[-1])]
                    loss_pose += self.pose_crit(mask, mask_label)
            loss_pose /= len(masks)
        
        loss += self.args.pose_f * loss_pose
        
        # accumulate gradient
        if phase == 'semi':
            loss /= 2.0

        loss.backward()
        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return {'loss': loss.item(), 
                'loss_pose': loss_pose.item(),
                'loss_ctc_m': 0.0 if loss_ctc_m is None else loss_ctc_m.item(),
                'loss_ctc_v': 0.0 if loss_ctc_v is None else loss_ctc_v.item(),
                'loss_dc_v2s': 0.0 if loss_dc_v2s is None else loss_dc_v2s.item(),
                'loss_dc_s2v': 0.0 if loss_dc_s2v is None else loss_dc_s2v.item(),
                'loss_sc': 0.0 if loss_sc is None else loss_sc.item()
                }
    
    
    def train(self):
        dtrain_labeled, dtrain_unlabeled = self.create_dataloader(split='train', bsize=self.args_model['batch_size'])
        dtrain_labeled_iter = enumerate(dtrain_labeled)
        labeled_loss_manager = LossManager(print_step=100)
        unlabeled_loss_manager = LossManager(print_step=100)
        self.model_manager = ModelManager(max_num_models=5)  # only save the best 3 models
        
        self.model.cuda()
        global_step = 0
        last_status = {'loss': -1., 'loss_trip': -1.}
        start_epoch = 0
        num_pretrain_epoch = self.args.num_pretrain_epoch if self.args.setting != 'semi_100' else self.args.max_num_epoch

        if self.args.pl:
            ckpt_file = os.path.join(self.args.save_dir, 'full_60.pkl')
            print('loading from {:s}'.format(ckpt_file))
            saved_dict = t.load(ckpt_file)
            self.model.load_state_dict(saved_dict['mainstream'])
        
        if self.args.from_ckpt:
            ckpt_file = os.path.join(self.args.save_dir, 'latest.pkl')
            print('loading from {:s}'.format(ckpt_file))
            saved_dict = t.load(ckpt_file)
            self.model.load_state_dict(saved_dict['mainstream'])
            self.optimizer.load_state_dict(saved_dict['optimizer'])
            self.lr_scheduler.load_state_dict(saved_dict['lr_scheduler'])
            start_epoch = saved_dict['epoch']+1
            if start_epoch < num_pretrain_epoch:
                global_step = self.num_labeled // self.args_model['batch_size'] * start_epoch
            else:
                global_step = self.num_labeled // self.args_model['batch_size'] * num_pretrain_epoch + \
                                self.num_unlabeled // self.args_model['batch_size'] * (start_epoch - num_pretrain_epoch)
            t.manual_seed(self.args.seed+start_epoch)  # change dataloader order
            np.random.seed(self.args.seed+start_epoch)
            dtrain_labeled, dtrain_unlabeled = self.create_dataloader(split='train', bsize=self.args_model['batch_size'])
            dtrain_labeled_iter = enumerate(dtrain_labeled)
        
        # train using labeled first
        for epoch in range(start_epoch, num_pretrain_epoch):
            epoch_loss = defaultdict(list)
            
            for i, batch_data in tqdm(enumerate(dtrain_labeled), desc='[Training with labeled, epoch {:d}]'.format(epoch)):
                global_step += 1
                loss_dict = self.train_batch(batch_data, epoch, phase='full', labeled=True, update=True)
                labeled_loss_manager.update(loss_dict, global_step)
                record_loss(loss_dict, epoch_loss)

                if self.args_lr_sch['patience'] == 6 and i == self.num_labeled//(self.args_model['batch_size']*2) and 'step' not in self.args_lr_sch['name']:  
                    #half of the epoch
                    self.validate(epoch, global_step)
            
            logging.info('Epoch: {:d}, loss: {:.3f} -> {:.3f}'.format(epoch, last_status['loss'], np.mean(epoch_loss['loss'])))
            last_status['loss'] = np.mean(epoch_loss['loss'])
            # tensorboard writer
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
            
            # break out by learning rate
            if self.lr_scheduler.optimizer.param_groups[0]["lr"] < 1e-5:
                break
        
        # train using both unlabeled and labeled
        if start_epoch < num_pretrain_epoch:
            start_epoch = num_pretrain_epoch
            epoch = num_pretrain_epoch - 1
            t.save({'mainstream': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch':epoch},
                    self.args.save_dir+'/full_{:d}.pkl'.format(num_pretrain_epoch))

        for epoch in range(start_epoch, self.args.max_num_epoch):
            labeled_epoch_loss = defaultdict(list)
            unlabeled_epoch_loss = defaultdict(list)
            
            for i, batch_data in tqdm(enumerate(dtrain_unlabeled), desc='[Training, epoch {:d}]'.format(epoch)):
                global_step += 1
                
                # unlabeled
                loss_dict = self.train_batch(batch_data, epoch, phase='semi', labeled=False, update=False)
                unlabeled_loss_manager.update(loss_dict, global_step)
                record_loss(loss_dict, unlabeled_epoch_loss)

                if self.args.pose is not None:
                    t.cuda.empty_cache()

                # labeled
                try:
                    _, batch_data = dtrain_labeled_iter.__next__()
                except:
                    dtrain_labeled_iter = enumerate(dtrain_labeled)
                    _, batch_data = dtrain_labeled_iter.__next__()
                loss_dict = self.train_batch(batch_data, epoch, phase='semi', labeled=True, update=True)
                labeled_loss_manager.update(loss_dict, global_step)
                record_loss(loss_dict, labeled_epoch_loss)
                
                if self.args.pose is not None:
                    t.cuda.empty_cache()
                
                if self.args_lr_sch['patience'] == 6 and i == self.num_unlabeled//(self.args_model['batch_size']*2) and 'step' not in self.args_lr_sch['name']:  
                    #half of the epoch
                    self.validate(epoch, global_step)
            
            logging.info('Epoch: {:d}, loss: {:.3f} -> {:.3f}'.format(epoch, last_status['loss'], np.mean(labeled_epoch_loss['loss'])))
            last_status['loss'] = np.mean(labeled_epoch_loss['loss'])
            
            # tensorboard writer
            for key in labeled_epoch_loss.keys():
                self.tb_writer.add_scalar('train/'+key, np.mean(labeled_epoch_loss[key]), global_step)
            for key in unlabeled_epoch_loss.keys():
                self.tb_writer.add_scalar('train_unlabeled/'+key, np.mean(unlabeled_epoch_loss[key]), global_step)
            
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
            
            # break out by learning rate
            if self.lr_scheduler.optimizer.param_groups[0]["lr"] < 1e-5:
                break
        
        self.tb_writer.close()
        os.remove(os.path.join(self.args.save_dir, 'latest.pkl'))  #finish and delete

    
    def train_ptf(self):
        dtrain_labeled, dtrain_unlabeled = self.create_dataloader(split='train', bsize=self.args_model['batch_size'])
        loss_manager = LossManager(print_step=100)
        self.model_manager = ModelManager(max_num_models=5)  # only save the best 3 models
        
        self.model.cuda()
        global_step = 0
        last_status = {'loss': -1., 'loss_trip': -1.}
        start_epoch = 0
        num_pretrain_epoch = self.args.num_pretrain_epoch if self.args.setting != 'semi_100' else self.args.max_num_epoch
        
        if self.args.from_ckpt:
            ckpt_file = os.path.join(self.args.save_dir, 'latest.pkl')
            print('loading from {:s}'.format(ckpt_file))
            saved_dict = t.load(ckpt_file)
            self.model.load_state_dict(saved_dict['mainstream'])
            self.optimizer.load_state_dict(saved_dict['optimizer'])
            self.lr_scheduler.load_state_dict(saved_dict['lr_scheduler'])
            start_epoch = saved_dict['epoch']+1
            if start_epoch < num_pretrain_epoch:
                global_step = self.num_labeled // self.args_model['batch_size'] * start_epoch
            else:
                global_step = self.num_labeled // self.args_model['batch_size'] * num_pretrain_epoch + \
                                self.num_unlabeled // self.args_model['batch_size'] * (start_epoch - num_pretrain_epoch)
            t.manual_seed(self.args.seed+start_epoch)  # change dataloader order
            np.random.seed(self.args.seed+start_epoch)
            dtrain_labeled, dtrain_unlabeled = self.create_dataloader(split='train', bsize=self.args_model['batch_size'])
        
        # pretrain using all data without label
        for epoch in range(start_epoch, num_pretrain_epoch):
            epoch_loss = defaultdict(list)
            
            for i, batch_data in tqdm(enumerate(dtrain_unlabeled), desc='[Pretraining with all, epoch {:d}]'.format(epoch)):
                global_step += 1
                loss_dict = self.train_batch(batch_data, epoch, phase='full', labeled=False, update=True)
                loss_manager.update(loss_dict, global_step)
                record_loss(loss_dict, epoch_loss)

                if self.args.pose is not None:
                    t.cuda.empty_cache()
            
            logging.info('Epoch: {:d}, loss: {:.3f} -> {:.3f}'.format(epoch, last_status['loss'], np.mean(epoch_loss['loss'])))
            last_status['loss'] = np.mean(epoch_loss['loss'])
            # tensorboard writer
            for key in epoch_loss.keys():
                self.tb_writer.add_scalar('train/'+key, np.mean(epoch_loss[key]), global_step)
            
            logging.info('--------------saving latest ckpt----------------')
            model_name = os.path.join(self.args.save_dir, 'latest.pkl')
            if os.path.exists(model_name):
                os.remove(model_name)
            t.save({'mainstream': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch':epoch},
                    model_name)
            
            self.lr_scheduler.step(last_status['loss'])
            
            # break out by learning rate
            if self.lr_scheduler.optimizer.param_groups[0]["lr"] < 1e-5:
                break
        
        if start_epoch < num_pretrain_epoch:
            start_epoch = num_pretrain_epoch
            epoch = num_pretrain_epoch - 1
            t.save({'mainstream': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch':epoch},
                    self.args.save_dir+'/pretrained.pkl')

        # train using both labeled
        # self.model.load_state_dict()
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()
        self.args.pose = None
        for epoch in range(start_epoch, self.args.max_num_epoch):
            epoch_loss = defaultdict(list)
            
            for i, batch_data in tqdm(enumerate(dtrain_labeled), desc='[Fine-tuning, epoch {:d}]'.format(epoch)):
                global_step += 1

                loss_dict = self.train_batch(batch_data, epoch, phase='full', labeled=True, update=True)
                loss_manager.update(loss_dict, global_step)
                record_loss(loss_dict, epoch_loss)
                
                if self.args_lr_sch['patience'] == 6 and i == self.num_labeled//(self.args_model['batch_size']*2) and 'step' not in self.args_lr_sch['name']:  
                    #half of the epoch
                    self.validate(epoch, global_step)
            
            logging.info('Epoch: {:d}, loss: {:.3f} -> {:.3f}'.format(epoch, last_status['loss'], np.mean(epoch_loss['loss'])))
            last_status['loss'] = np.mean(epoch_loss['loss'])
            
            # tensorboard writer
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
            
            # break out by learning rate
            if self.lr_scheduler.optimizer.param_groups[0]["lr"] < 1e-5:
                break
        
        self.tb_writer.close()
        os.remove(os.path.join(self.args.save_dir, 'latest.pkl'))  #finish and delete

