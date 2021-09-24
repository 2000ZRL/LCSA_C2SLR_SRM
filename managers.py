# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:40:23 2020

@author: Ronglai ZUO
Script for training managers
"""

import torch as t

t.backends.cudnn.deterministic = True

import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence

import numpy as np
import os
import logging
# import uuid

from modules.dual import Dual_v0, Dual_v2
# from signjoey.embeddings import MaskedMean
from utils.utils import LossManager, ModelManager, freeze_params, unfreeze_params
# from model import Simple_SLP, FCN_mainstream
# from evaluation_relaxation.phoenix_eval import get_phoenix_wer
from utils.metric import get_wer_delsubins
# from phoenix_datasets.datasets import load_pil

from itertools import groupby
from tqdm import tqdm
from base import TrainingManager


class TrainingManager_old(TrainingManager):
    def __init__(self, args, vocab):
        super(TrainingManager_old, self).__init__(args, vocab)
        self.optimizer = self._create_optimizer(self.model.parameters())
        self.lr_scheduler = self._create_lr_scheduler()
        

class TrainingManager_Dual_v0(TrainingManager):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.decoded_proposal = {}
        if bool(args.gfe):
            self.gfe = Dual_v0(emb_size=512,
                               fea_size=512,
                               voc_size=self.voc_size,
                               seq_model_type=args.seq_model_type,
                               sen_level_fea_type=args.sen_level_fea_type,
                               use_se=False,
                               freeze_sen_level_encoder=False)
            self.optimizer = self._create_optimizer([{'params': self.model.parameters()},
                                                     {'params': self.gfe.seq_model.parameters()}])
            if self.args.gfe_cri == 'mse':
                self.gfe_criterion = nn.MSELoss().cuda()
            elif self.args.gfe_cri == 'cos':
                self.gfe_criterion = nn.CosineEmbeddingLoss().cuda()
                self.cos_target_real = t.ones(self.args.batch_size).cuda()
        
            self.lr_scheduler = self._create_lr_scheduler()
        
    def update_proposal(self, epoch):
        dtrain_no_drop = self.create_dataloader(split='train_no_drop', bsize=1)
        logging.info('--------Updating decoded gloss proposal, epoch: {:d} ----------'.format(epoch))
        for i, batch_data in tqdm(enumerate(dtrain_no_drop), desc='[Updating, epoch {:d}]'.format(epoch)):
            self.eval_batch(batch_data, for_train=True)
    
    def get_src_dual_fea(self, gls_scores, len_video, guide_fea, video_id, epoch):
        #just decoding
        gls_scores = F.softmax(gls_scores, dim=-1)
        pred_seq, _, _, out_seq_len = self.ctc_decoder.decode(gls_scores, len_video)
        hyp = []
        len_hyp = []
        if self.args.update_proposal_freq == 0 or\
            (epoch - self.start_gfe_epoch) % self.args.update_proposal_freq == 0:
            for i in range(len_video.shape[0]):
                res = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                if res == []:
                    #avoid decoded results are empty. 
                    #But in fact, this will not happen after several epochs
                    hyp.extend([1,1,1])
                    len_hyp.append(3)
                else:
                    hyp.extend(res)
                    len_hyp.append(len(res))
                    if self.args.update_proposal_freq > 0:
                        self.decoded_proposal[''.join(video_id[i])] = res
        else:
            for i in range(len_video.shape[0]):
                hyp.extend(self.decoded_proposal[''.join(video_id[i])])
                len_hyp.append(len(self.decoded_proposal[''.join(video_id[i])]))
        
        # hyp = pad_sequence(hyp, batch_first=True, padding_value=self.voc_size-2).long().cuda()
        # len_hyp = t.LongTensor(len_hyp).cuda()
        
        return self.gfe(t.LongTensor(hyp).cuda(), t.LongTensor(len_hyp), guide_fea)
    
    
    def train_batch(self, batch_data, epoch, phase='main', need_which_level=[1], need_src=False, need_gt=True):
        video = t.cat(batch_data['video']).cuda()
        len_video_ori = batch_data['len_video'].cuda()
        label = batch_data['label'].cuda()
        len_label = batch_data['len_label'].cuda()
        if need_src:
            video_id = batch_data['id']
        
        self.model.train()
        gls_scores, len_video, frame_fea, gls_fea = self.model(video, len_video_ori, need_which_level)
        
        gls_scores = gls_scores.log_softmax(-1)
        gls_scores = gls_scores.permute(1,0,2)  #[max_T, B, C]
        loss = self.criterion(gls_scores, label, len_video, len_label)
        # loss = loss.mean()
        gfe_loss_from_gt = None
        gfe_loss_from_src = None
        gfe_loss = None
        
        if bool(self.args.gfe) and epoch >= self.args.start_gfe_epoch:
            self.gfe.train()
            
            #for guide feature
            idx = np.random.choice(min(len_video_ori.cpu()), size=1)[0]
            
            #for pretraining gfe module
            if phase == 'pretrain':
                if 1 in need_which_level:
                    gls_fea = gls_fea.detach()
                    gls_fea.requires_grad = True
                elif 0 in need_which_level:
                    frame_fea = frame_fea.detach()
                    frame_fea.requires_grad = True
            
            #generate dual features and dual label from gt gloss
            if need_gt:
                if 1 in need_which_level:
                    gt_dual_fea, dual_fea_label = self.gfe(label, 
                                                           len_label,
                                                           gls_fea,
                                                           len_video,
                                                           guide_fea=None)
                elif 0 in need_which_level:
                    gt_dual_fea, dual_fea_label = self.gfe(label, 
                                                           len_label,
                                                           frame_fea,
                                                           len_video_ori,
                                                           guide_fea=frame_fea[..., idx].detach())
            
            #generate dual features from decoded gloss, if don't need gt we should also generate dual_label
            if need_src:
                gls = []
                len_gls = []
                for id in video_id:
                    id = ''.join(id)
                    gls.extend(self.decoded_proposal[id])
                    len_gls.append(len(self.decoded_proposal[id]))
                gls = t.LongTensor(gls).cuda()
                len_gls = t.LongTensor(len_gls)
                
                if not need_gt:
                    if 1 in need_which_level:
                        src_dual_fea, dual_fea_label = self.gfe(gls, 
                                                                len_gls,
                                                                gls_fea,
                                                                len_video,
                                                                guide_fea=None)
                    elif 0 in need_which_level:
                        src_dual_fea, dual_fea_label = self.gfe(gls, 
                                                                len_gls,
                                                                frame_fea,
                                                                len_video_ori,
                                                                guide_fea=frame_fea[..., idx].detach())
                else:
                    if 1 in need_which_level:
                        src_dual_fea, _ = self.gfe(gls, 
                                                   len_gls,
                                                   None,
                                                   None,
                                                   guide_fea=None)
                    elif 0 in need_which_level:
                        src_dual_fea, _ = self.gfe(gls, 
                                                   len_gls,
                                                   None,
                                                   None,
                                                   guide_fea=frame_fea[..., idx].detach())
            
            #compute gfe_loss from decoded gloss
            if need_src:
                assert src_dual_fea.shape[:] == dual_fea_label.shape[:]
                if self.args.gfe_cri == 'mse':
                    gfe_loss_from_src = self.gfe_criterion(src_dual_fea, dual_fea_label)
                elif self.args.gfe_cri == 'cos':
                    gfe_loss_from_src = self.gfe_criterion(src_dual_fea, dual_fea_label, self.cos_target_real)
            
            #compute gfe_loss from gt gloss
            if need_gt:
                assert gt_dual_fea.shape[:] == dual_fea_label.shape[:]
                if self.args.gfe_cri == 'mse':
                    gfe_loss_from_gt = self.gfe_criterion(gt_dual_fea, dual_fea_label)
                elif self.args.gfe_cri == 'cos':
                    gfe_loss_from_gt = self.gfe_criterion(gt_dual_fea, dual_fea_label, self.cos_target_real)
            
            if need_src and need_gt:
                gfe_loss = self.args.weight_label * gfe_loss_from_gt + \
                            (1-self.args.weight_label) * gfe_loss_from_src
            elif need_src and not need_gt:
                gfe_loss = gfe_loss_from_src
            elif need_gt and not need_src:
                gfe_loss = gfe_loss_from_gt

            loss += self.args.lambda_gfe * gfe_loss
        
        if phase == 'main':
            loss.backward()
        elif phase == 'pretrain':
            gfe_loss.backward()
        else:
            raise ValueError('We only support main and pretrain for gfe phase!\n')
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        if gfe_loss is not None:
            return loss.item(), gfe_loss.item()
        else:
            return loss.item(), None
    
    
    def train(self):
        dtrain = self.create_dataloader(split='train', bsize=self.args.batch_size)
        loss_manager = LossManager(print_step=100)
        self.model_manager = ModelManager(max_num_models=5)  #only save the best 5 models
        
        self.model.cuda()
        if bool(self.args.gfe):
            self.gfe.cuda()
            freeze_params(self.gfe)
        
        max_num_epoch = self.args.max_num_epoch
        global_step = 0
        gfe_step = 0
        last_status = {'loss': -1., 'gfe_loss': -1., 'loss_trip': -1.}
        start_epoch = 0
        self.start_gfe_epoch = self.args.start_gfe_epoch
        
        if bool(self.args.from_ms_ckpt):
            logging.info('-----------loading from mainstream ckpt {:s}-------'.format(self.args.ms_ckpt_file))
            dic = t.load(self.args.ms_ckpt_file)
            self.model.load_state_dict(dic['mainstream'], strict=bool(self.args.strict))
            #there would be bug if loading optimizer and lr_scheduler
            #vim /2tssd/rzuo/anaconda3/lib/python3.8/site-packages/torch/optim/lr_scheduler.py
            # self.optimizer.load_state_dict(dic['optimizer'])
            # self.lr_scheduler.load_state_dict(dic['lr_scheduler'])
            if 'epoch' not in list(dic.keys()):
                start_epoch = self.start_gfe_epoch
            else:
                start_epoch = dic['epoch']+1
        
        if bool(self.args.from_ckpt):
            logging.info('-----------loading from ckpt {:s}-------'.format(self.args.ckpt_file))
            dic = t.load(self.args.ckpt_file)
            self.model.load_state_dict(dic['mainstream'], strict=True)
            self.gfe.load_state_dict(dic['gfe'], strict=True)
            self.optimizer.load_state_dict(dic['optimizer'])
            self.lr_scheduler.load_state_dict(dic['lr_scheduler'])
            start_epoch = dic['epoch']+1
            if start_epoch > self.args.start_gfe_epoch:
                unfreeze_params(self.gfe)
                self.start_gfe_epoch = start_epoch
        
        for epoch in range(start_epoch, max_num_epoch):
            #*********************Training*******************
            epoch_loss = []
            epoch_gfe_loss = []
            
            if epoch == 15:
                ms_dir = os.path.join(self.args.save_dir, 'mainstream_ckpt')
                if not os.path.exists(ms_dir):
                    os.makedirs(ms_dir)
                model_name = os.path.join(ms_dir, 'ep{:d}.pkl'.format(self.args.start_gfe_epoch-1))
                logging.info('-----------saving mainstream ckpt to {:s}-------'.format(model_name))
                t.save({'mainstream': self.model.state_dict(), 
                        'optimizer': self.optimizer.state_dict()}, 
                        model_name)
            
            if bool(self.args.gfe) and \
                bool(self.args.need_src) and \
                epoch >= self.start_gfe_epoch and \
                self.args.update_proposal_freq > 0 and \
                (epoch - self.start_gfe_epoch) % self.args.update_proposal_freq == 0:
                self.update_proposal(epoch)
                # self.decoded_proposal['1'] = [x for x in range(28)]
                # self.decoded_proposal['2'] = [x for x in range(27)]
            
            if bool(self.args.gfe) and epoch == self.args.start_gfe_epoch:
                # ms_dir = os.path.join(self.args.save_dir, 'mainstream_ckpt')
                # if not os.path.exists(ms_dir):
                #     os.makedirs(ms_dir)
                # model_name = os.path.join(ms_dir, 'ep{:d}.pkl'.format(self.args.start_gfe_epoch-1))
                # logging.info('-----------saving mainstream ckpt to {:s}-------'.format(model_name))
                # t.save({'mainstream': self.model.state_dict(), 
                #         'optimizer': self.optimizer.state_dict()}, 
                #         model_name)
                unfreeze_params(self.gfe)
                for ep_pre in range(self.args.num_pretrain_epoch):
                    logging.info('-------Pretraining for GFE: epoch {:d} of {:d}-------'.format(\
                        ep_pre, self.args.num_pretrain_epoch))
                    for i, batch_data in tqdm(enumerate(dtrain), desc='[Pretraining, epoch {:d}]'.format(ep_pre)):
                        gfe_step += 1
                        loss, gfe_loss = self.train_batch(batch_data, 
                                                          epoch, 
                                                          phase='pretrain',
                                                          need_which_level=[self.args.need_which_level],
                                                          need_src=bool(self.args.need_src),
                                                          need_gt=bool(self.args.need_gt))
                        loss_manager.update(loss, gfe_loss, gfe_step)
            
            # batch_data = {'video': [t.rand(299,3,224,224), t.rand(296,3,224,224)],
            #               'len_video': t.LongTensor([299,296]),
            #               'label': t.LongTensor(np.random.randint(low=0, high=1231, size=55)),
            #               'len_label': t.LongTensor([28,27]),
            #               'id': [['1'], ['2']]}
            
            # for i, batch_data in enumerate([batch_data]):
            for i, batch_data in tqdm(enumerate(dtrain), desc='[Training, epoch {:d}]'.format(epoch)):
                global_step += 1
                
                loss, gfe_loss = self.train_batch(batch_data, 
                                                  epoch,
                                                  phase='main',
                                                  need_which_level=[self.args.need_which_level],
                                                  need_src=bool(self.args.need_src),
                                                  need_gt=bool(self.args.need_gt))
                
                if gfe_loss is not None:
                    epoch_gfe_loss.append(gfe_loss)
                
                loss_manager.update(loss, gfe_loss, global_step)
                epoch_loss.append(loss)
                
                if i == 5672//(self.args.batch_size*2):  #half of the epoch
                    self.validate(epoch, global_step)
            
            logging.info('Epoch: {:d}, loss: {:.5f} -> {:.5f}'.format(epoch, last_status['loss'], np.mean(epoch_loss)))
            self.tb_writer.add_scalar('train/training_total_loss', np.mean(epoch_loss), global_step,)
            last_status['loss'] = np.mean(epoch_loss)
            if bool(self.args.gfe) and epoch >= self.args.start_gfe_epoch:
                logging.info('Epoch: {:d}, gfe_loss: {:.5f} -> {:.5f}'.format(\
                    epoch, last_status['gfe_loss'], np.mean(epoch_gfe_loss)))
                self.tb_writer.add_scalar('train/gfe_loss', np.mean(epoch_gfe_loss), global_step,)
                last_status['gfe_loss'] = np.mean(epoch_gfe_loss)
        
            self.validate(epoch, global_step)
            
            logging.info('--------------saving latest ckpt----------------')
            model_name = os.path.join(self.args.save_dir, 'latest.pkl')
            if os.path.exists(model_name):
                os.remove(model_name)
            t.save({'mainstream': self.model.state_dict(),
                    'gfe': self.gfe.state_dict() if bool(self.args.gfe) else None,
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch':epoch},
                    model_name)
        
        self.tb_writer.close()


class TrainingManager_Dual_v2(TrainingManager_Dual_v0):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.decoded_proposal = {}
        if bool(args.gfe):
            self.gfe = Dual_v2(emb_size=512,
                               hidden_size=128,
                               voc_size=self.voc_size,
                               seq_model_type=args.seq_model_type
                               )
            self.optimizer = self._create_optimizer([{'params': self.model.parameters()},
                                                     {'params': self.gfe.parameters()}])

            self.gfe_criterion = nn.CosineEmbeddingLoss().cuda()
            self.cos_target_real = t.ones(self.args.batch_size).cuda()
        
        self.lr_scheduler = self._create_lr_scheduler()
    
    
    def train_batch(self, batch_data, epoch, phase='main', need_which_level=[1], need_src=False, need_gt=True):
        video = t.cat(batch_data['video']).cuda()
        len_video_ori = batch_data['len_video'].cuda()
        label = batch_data['label'].cuda()
        len_label = batch_data['len_label'].cuda()
        if need_src:
            video_id = batch_data['id']
        
        self.model.train()
        gls_scores, len_video, frame_fea, gls_fea = self.model(video, len_video_ori, need_which_level)
        
        gls_scores = gls_scores.log_softmax(-1)
        gls_scores = gls_scores.permute(1,0,2)  #[max_T, B, C]
        loss = self.criterion(gls_scores, label, len_video, len_label)
        # loss = loss.mean()
        gfe_loss_from_gt = None
        gfe_loss_from_src = None
        gfe_loss = None
        
        if bool(self.args.gfe) and epoch >= self.args.start_gfe_epoch:
            self.gfe.train()
            
            #for guide feature
            idx = np.random.choice(min(len_video_ori.cpu()), size=1)[0]
            
            #for pretraining gfe module
            if phase == 'pretrain':
                if 1 in need_which_level:
                    gls_fea = gls_fea.detach()
                    gls_fea.requires_grad = True
                elif 0 in need_which_level:
                    frame_fea = frame_fea.detach()
                    frame_fea.requires_grad = True
            
            #generate dual features and dual label from gt gloss
            if need_gt:
                if 1 in need_which_level:
                    gt_gls_before_mlp, gt_gls, fea_before_mlp, fea = self.gfe(label, 
                                                                              len_label,
                                                                              gls_fea,
                                                                              len_video,
                                                                              guide_fea=None)
                elif 0 in need_which_level:
                    gt_gls_before_mlp, gt_gls, fea_before_mlp, fea = self.gfe(label, 
                                                                              len_label,
                                                                              frame_fea,
                                                                              len_video_ori,
                                                                              guide_fea=None)
            
            #generate dual features from decoded gloss, if don't need gt we should also generate dual_label
            if need_src:
                gls = []
                len_gls = []
                for id in video_id:
                    id = ''.join(id)
                    gls.extend(self.decoded_proposal[id])
                    len_gls.append(len(self.decoded_proposal[id]))
                gls = t.LongTensor(gls).cuda()
                len_gls = t.LongTensor(len_gls)
                
                if not need_gt:
                    if 1 in need_which_level:
                        dec_gls_before_mlp, dec_gls, fea_before_mlp, fea = self.gfe(gls, 
                                                                                    len_gls,
                                                                                    gls_fea,
                                                                                    len_video,
                                                                                    guide_fea=None)
                    elif 0 in need_which_level:
                        dec_gls_before_mlp, dec_gls, fea_before_mlp, fea = self.gfe(gls, 
                                                                                    len_gls,
                                                                                    frame_fea,
                                                                                    len_video_ori,
                                                                                    guide_fea=None)
                else:
                    if 1 in need_which_level:
                        dec_gls_before_mlp, dec_gls, _, _ = self.gfe(gls, 
                                                                     len_gls,
                                                                     gls_fea,
                                                                     len_video,
                                                                     guide_fea=None)
                    elif 0 in need_which_level:
                        dec_gls_before_mlp, dec_gls, _, _ = self.gfe(gls, 
                                                                     len_gls,
                                                                     frame_fea,
                                                                     len_video_ori,
                                                                     guide_fea=None)
            
            #compute gfe_loss from decoded gloss
            if need_src:
                assert dec_gls_before_mlp.shape[:] == fea.shape[:]
                assert dec_gls.shape[:] == fea_before_mlp.shape[:]
                gfe_loss_from_src = 0.5*self.gfe_criterion(fea, dec_gls_before_mlp.detach(), self.cos_target_real)+\
                    0.5*self.gfe_criterion(dec_gls, fea_before_mlp.detach(), self.cos_target_real)
            
            #compute gfe_loss from gt gloss
            if need_gt:
                assert gt_gls_before_mlp.shape[:] == fea.shape[:]
                assert gt_gls.shape[:] == fea_before_mlp.shape[:]
                gfe_loss_from_gt = 0.5*self.gfe_criterion(fea, gt_gls_before_mlp.detach(), self.cos_target_real)+\
                    0.5*self.gfe_criterion(gt_gls, fea_before_mlp.detach(), self.cos_target_real)
            
            if need_src and need_gt:
                gfe_loss = self.args.weight_label * gfe_loss_from_gt + \
                            (1-self.args.weight_label) * gfe_loss_from_src
            elif need_src and not need_gt:
                gfe_loss = gfe_loss_from_src
            elif need_gt and not need_src:
                gfe_loss = gfe_loss_from_gt

            loss += self.args.lambda_gfe * gfe_loss
        
        if phase == 'main':
            loss.backward()
        elif phase == 'pretrain':
            gfe_loss.backward()
        else:
            raise ValueError('We only support main and pretrain for gfe phase!\n')
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        if gfe_loss is not None:
            return loss.item(), gfe_loss.item()
        else:
            return loss.item(), None


class TrainingManager_FCN_sen(TrainingManager_Dual_v0):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.decoded_proposal = {}
        self.optimizer = self._create_optimizer(params=self.model.parameters())

        self.gfe_criterion = nn.CosineEmbeddingLoss().cuda()
        self.cos_target_real = t.ones(self.args.batch_size).cuda()
        
        self.lr_scheduler = self._create_lr_scheduler()
    
    
    def eval_batch(self, batch_data, for_train=False):
        with t.no_grad():
            batch_size = len(batch_data['video'])
            video = t.cat(batch_data['video']).cuda()
            len_video = batch_data['len_video'].cuda()
            label = batch_data['label'].cuda()
            len_label = batch_data['len_label'].cuda()
            video_id = batch_data['id']
            
            
            self.model.eval()
            gls_scores, len_video, _, _, _, _ = self.model(video, len_video)
            
            #compute validaiton loss
            gls_prob = F.log_softmax(gls_scores, -1)
            gls_prob = gls_prob.permute(1,0,2)
            val_loss = self.criterion(gls_prob, label, len_video, len_label)
            
            #ctc decode
            gls_scores = F.softmax(gls_scores, dim=-1)
            pred_seq, beam_scores, _, out_seq_len = self.ctc_decoder.decode(gls_scores, len_video)
            
            #metrics evaluation: wer
            assert pred_seq.shape[0] == batch_size
            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            start = 0
            for i, length in enumerate(len_label):
                end = start + length
                ref = label[start:end].tolist()
                if self.lm is None:
                    hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                else:
                    hyp = self.get_hyp_after_lm(pred_seq[i], beam_scores[i], out_seq_len[i])
                    
                if not for_train:
                    self.decoded_dict[''.join(video_id[i])] = hyp
                else:
                    self.decoded_proposal[''.join(video_id[i])] = hyp
                    
                correct += int(ref == hyp)
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1
                start = end
            assert end == label.size(0)
        
        return err_delsubins, correct, count, val_loss.item()
    
    
    def train_batch(self, batch_data, epoch, phase='main', need_which_level=[1], need_src=False, need_gt=True):
        video = t.cat(batch_data['video']).cuda()
        len_video_ori = batch_data['len_video'].cuda()
        label = batch_data['label'].cuda()
        len_label = batch_data['len_label'].cuda()
        if need_src:
            video_id = batch_data['id']
        
        self.model.train()
        if need_gt:
            gls = label
            len_gls = len_label
            
        elif need_src:
            gls = []
            len_gls = []
            for id in video_id:
                id = ''.join(id)
                gls.extend(self.decoded_proposal[id])
                len_gls.append(len(self.decoded_proposal[id]))
            gls = t.LongTensor(gls).cuda()
            len_gls = t.LongTensor(len_gls)
        
        gls_scores, len_video, gls_before_pred, gls_after_pred, fea_before_pred, fea_after_pred = \
            self.model(video, len_video_ori, label, len_label, bool(self.args.train_rep))
        
        gls_scores = gls_scores.log_softmax(-1)
        gls_scores = gls_scores.permute(1,0,2)  #[max_T, B, C]
        loss = self.criterion(gls_scores, label, len_video, len_label)
        # loss = loss.mean()
        gfe_loss_from_gt = None
        gfe_loss_from_src = None
        gfe_loss = None
        
        if bool(self.args.gfe) and epoch >= self.args.start_gfe_epoch:
            #for guide feature
            # idx = np.random.choice(min(len_video_ori.cpu()), size=1)[0]
            
            #compute gfe_loss from gt gloss
            if self.args.train_rep:
                assert gls_before_pred.shape[:] == fea_after_pred.shape[:]
                assert gls_after_pred.shape[:] == fea_before_pred.shape[:]
                gfe_loss = 0.5*self.gfe_criterion(fea_after_pred, gls_before_pred.detach(), self.cos_target_real)+\
                    0.5*self.gfe_criterion(gls_after_pred, fea_before_pred.detach(), self.cos_target_real)
                loss += self.args.lambda_gfe * gfe_loss
        
        if phase == 'main':
            loss.backward()
        elif phase == 'pretrain':
            gfe_loss.backward()
        else:
            raise ValueError('We only support main and pretrain for gfe phase!\n')
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        if gfe_loss is not None:
            return loss.item(), gfe_loss.item()
        else:
            return loss.item(), None


class TrainingManager_align(TrainingManager_Dual_v0):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.optimizer = self._create_optimizer(self.model.parameters())
        self.lr_scheduler = self._create_lr_scheduler()
        self.length_criterion = nn.MSELoss().cuda()
        self.max_gls_len = 28
        
    
    def eval_batch(self, batch_data, for_train=False):
        with t.no_grad():
            batch_size = len(batch_data['video'])
            video = t.cat(batch_data['video']).cuda()
            len_video = batch_data['len_video'].cuda()
            label = batch_data['label'].cuda()
            len_label = batch_data['len_label'].cuda()
            video_id = batch_data['id']
            
            self.model.eval()
            gls_scores, len_video, _, _ = self.model(video, len_video)
            len_video = t.clamp(len_video, 1, self.max_gls_len).long()
            
            #compute validaiton loss
            gls_prob = F.log_softmax(gls_scores, -1)
            gls_prob = gls_prob.permute(1,0,2)
            val_loss = self.criterion(gls_prob, label, len_video, len_label)
            
            #ctc decode
            gls_scores = F.softmax(gls_scores, dim=-1)
            pred_seq, beam_scores, _, out_seq_len = self.ctc_decoder.decode(gls_scores, len_video)
            
            #metrics evaluation: wer
            assert pred_seq.shape[0] == batch_size
            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            start = 0
            for i, length in enumerate(len_label):
                end = start + length
                ref = label[start:end].tolist()
                if self.lm is None:
                    hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                else:
                    hyp = self.get_hyp_after_lm(pred_seq[i], beam_scores[i], out_seq_len[i])
                    
                if not for_train:
                    self.decoded_dict[''.join(video_id[i])] = hyp
                else:
                    self.decoded_proposal[''.join(video_id[i])] = hyp
                    
                correct += int(ref == hyp)
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1
                start = end
            assert end == label.size(0)
        
        return err_delsubins, correct, count, val_loss.item()
    
    
    def train_batch(self, batch_data, epoch, phase='main', need_which_level=[1], need_src=False, need_gt=False):
        video = t.cat(batch_data['video']).cuda()
        len_video = batch_data['len_video'].cuda()
        label = batch_data['label'].cuda()
        len_label = batch_data['len_label'].cuda()
        
        self.model.train()
        gls_scores, len_video, _, _ = self.model(video, len_video)
        
        #lenght loss
        loss_len = 0.5*self.length_criterion(len_video, len_label.float())
        
        #ctc decode
        gls_prob = gls_scores.log_softmax(-1)
        gls_prob = gls_prob.permute(1,0,2)
        len_video_clamp = t.clamp(len_video, 1, self.max_gls_len).long()
        loss = self.criterion(gls_prob, label, len_video_clamp, len_label)
        loss = loss.mean()
        loss += self.args.lambda_gfe * loss_len   #lambda_gfe means length_weight here
        
        loss.backward()
        # loss_len.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), loss_len.item()
