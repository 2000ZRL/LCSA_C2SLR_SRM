# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:11:06 2020

@author: Ronglai ZUO
"""

import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tfmer.embeddings import Embeddings, MaskedMean, MaskedNorm
from tfmer.encoders import TransformerEncoder, RecurrentEncoder
from .tcn import TCN_block
from .se import SE_block_time


class Dual_v0(nn.Module):
    def __init__(self, 
                 emb_size=512, 
                 fea_size=512, 
                 voc_size=1233, 
                 seq_model_type='tcn', 
                 seq_model_layer=1,
                 sen_level_fea_type='last', 
                 use_se=False,
                 freeze_sen_level_encoder=False):
        #seq_model for encode gloss. sen_level_encoder for encode feature from mainstream
        super(Dual_v0, self).__init__()
        self.se = use_se
        self.padding_index = voc_size-2  #unk
        self.seq_model_type = seq_model_type
        self.sen_level_fea_type = sen_level_fea_type
        self.emb = Embeddings(embedding_dim=emb_size,
                              num_heads=1,
                              norm_type='batch',
                              activation_type='relu',
                              vocab_size=voc_size,
                              padding_index=self.padding_index)

        if seq_model_type == 'tcn':
            self.seq_model = TCN_block(inchannels=emb_size,
                                 outchannels=fea_size,
                                 kernel_size=3,
                                 stride=1,
                                 norm_type='batch',
                                 use_pool=False)
        elif seq_model_type == 'transformer':
            self.seq_model = TransformerEncoder(hidden_size=emb_size,
                                                ff_size=2048,
                                                num_layers=seq_model_layer,
                                                num_heads=4,
                                                dropout=0.1,
                                                emb_dropout=0.1,
                                                freeze=False)
        elif seq_model_type == 'gru' or seq_model_type == 'lstm':
            self.seq_model = RecurrentEncoder(rnn_type=seq_model_type,
                                              hidden_size=fea_size,
                                              emb_size=emb_size,
                                              num_layers=seq_model_layer,
                                              dropout=0.1,
                                              emb_dropout=0.1,
                                              bidirectional=True)
        else:
            raise ValueError('We only support tcn, transformer, gru and lstm now.\n')
            
        if sen_level_fea_type == 'last':
            self.sen_level_encoder = RecurrentEncoder(rnn_type=seq_model_type,
                                                      hidden_size=fea_size,
                                                      emb_size=emb_size,
                                                      num_layers=1,
                                                      dropout=0.1,
                                                      emb_dropout=0.1,
                                                      bidirectional=True,
                                                      freeze=freeze_sen_level_encoder)
        elif sen_level_fea_type == 'mean':
            self.sen_level_encoder = MaskedMean
        
        if self.se:
            self.SE_block = SE_block_time(input_size=fea_size,
                                          ratio=8,
                                          act_fun='relu')
        
    def forward(self, gloss, len_gloss, level_fea, len_fea, guide_fea=None):
        '''
        Parameters
        ----------
        gloss : cuda Tensor
            shape [sum_T]
        len_gloss : cuda LongTensor
            shape [B]
        level_fea : cuda Tensor, frame-level (0-level) or gloss-level (1-level)
            shape [B,C,T]
        guide_fea : cuda Tensor, feature of one sample in the batch
            shape [B,C]

        Returns
        -------
        sentence-level representation via average pooling
        shape [B,C]
        '''
        #gloss
        gloss = gloss.split(len_gloss.tolist())
        gloss = pad_sequence(gloss, batch_first=True, padding_value=self.padding_index).long()
        
        mask = t.zeros(len_gloss.shape[0], 1, max(len_gloss)).bool().cuda()
        for i, l in enumerate(len_gloss):
            mask[i, 0, l:] = True
        
        gloss = self.emb(gloss, mask)  #[B,C,max_T]
        if guide_fea is not None:
            gloss = gloss.permute(2,0,1)
            gloss += guide_fea  #or concat?
            gloss = gloss.permute(1,2,0)
        
        if self.seq_model_type == 'tcn':
            gloss = self.seq_model(gloss, mask)
        elif self.seq_model_type == 'transformer':
            gloss, _ = self.seq_model(gloss.transpose(1,2), len_gloss, mask)
            gloss = gloss.transpose(1,2)  #[B,C,T]
        elif self.seq_model_type == 'gru' or self.seq_model_type == 'lstm':
            _, gloss = self.seq_model(gloss.transpose(1,2), len_gloss, True) #[B,C], only need hidden
        
        if self.se:
            gloss = self.SE_block(gloss, mask, need_mid=True)
        
        #compute average only for the len_gloss
        if self.seq_model_type != 'gru' and self.seq_model_type != 'lstm':
            gloss = MaskedMean(gloss, mask)
            
        #sentence-level feature, make dual features label
        if level_fea is not None:
            if self.sen_level_fea_type == 'mean':
                mask = t.zeros(len_fea.shape[0], 1, max(len_fea)).bool().cuda()
                for i, l in enumerate(len_fea):
                    mask[i, 0, l:] = True
                dual_fea_label = self.sen_level_encoder(level_fea, mask)  #[B,C]
            elif self.sen_level_fea_type == 'last':
                _, dual_fea_label = self.sen_level_encoder(level_fea.transpose(1,2), len_fea, True)  #[B,C]
        else:
            dual_fea_label = None

        return gloss, dual_fea_label


class Dual_v2(nn.Module):
    #Similar to SimSiam
    def __init__(self, emb_size=512, hidden_size=128, voc_size=1233, seq_model_type='gru'):
        super(Dual_v2, self).__init__()
        self.padding_index = voc_size-2  #unk
        self.emb = Embeddings(embedding_dim=emb_size,
                              num_heads=1,
                              norm_type='batch',
                              activation_type='relu',
                              vocab_size=voc_size,
                              padding_index=self.padding_index)
        
        if seq_model_type == 'gru' or seq_model_type == 'lstm':
            self.seq_model = RecurrentEncoder(rnn_type=seq_model_type,
                                              hidden_size=emb_size//2,
                                              emb_size=emb_size,
                                              num_layers=1,
                                              dropout=0.1,
                                              emb_dropout=0.1,
                                              bidirectional=True)
        
        self.projector = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(emb_size, emb_size),
            )
        
        self.predictor = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(emb_size, emb_size)
            )
        
    
    def forward(self, gls, len_gls, fea, len_fea, guide_fea=None):
        #generate gloss embedding
        gls = gls.split(len_gls.tolist())
        gls = pad_sequence(gls, batch_first=True, padding_value=self.padding_index).long()
        
        mask = t.zeros(len_gls.shape[0], 1, max(len_gls)).bool().cuda()
        for i, l in enumerate(len_gls):
            mask[i, 0, l:] = True
        
        gls = self.emb(gls, mask)  #[B,C,max_T]
        
        #forward gls
        #maybe we should also need output to make batch size larger
        _, gls = self.seq_model(gls.transpose(1,2), len_gls, True) #[B,2C], only need hidden last
        gls = self.projector(gls)
        gls_before_mlp = gls
        gls = self.predictor(gls)
        
        #forward level feature (frame-level or gloss level)
        _, fea = self.seq_model(fea.transpose(1,2), len_fea, True) #[B,2C], only need hidden last
        fea = self.projector(fea)
        fea_before_mlp = fea
        fea = self.predictor(gls)
        
        return gls_before_mlp, gls, fea_before_mlp, fea


class Dual_v1(nn.Module):
    def __init__(self, frame_fea_size=512, gls_fea_size=512, num_layers=2, seq_model_type='transformer', use_se=False):
        super(Dual_v1, self).__init__()
        self.use_se = use_se
        self.seq_model_type = seq_model_type
        
        if self.seq_model_type == 'transformer':
            self.seq_model = TransformerEncoder(hidden_size=gls_fea_size,
                                                ff_size=2048,
                                                num_layers=num_layers,
                                                num_heads=4,
                                                dropout=0.1,
                                                emb_dropout=0.1,
                                                freeze=False)
        elif self.seq_model_type == 'tcn':
            self.seq_model = nn.Sequential(TCN_block(inchannels=gls_fea_size,
                                                     outchannels=frame_fea_size,
                                                     kernel_size=3,
                                                     stride=1,
                                                     norm_type='batch',
                                                     use_pool=False),
                                           TCN_block(inchannels=gls_fea_size,
                                                     outchannels=frame_fea_size,
                                                     kernel_size=3,
                                                     stride=1,
                                                     norm_type='batch',
                                                     use_pool=False))
            
        if self.use_se:
            self.SE_block = SE_block_time(input_size=gls_fea_size,
                                          ratio=8,
                                          act_fun='relu')
            
    def forward(self, gls_fea, len_gls, guide_fea, frame_fea=None):
        '''
        Parameters
        ----------
        frame_fea : cuda Tensor
            shape [B,C,T]
        gls_fea : cuda Tensor:
            shape [B,C,T']
        len_gls : cuda LongTensor
            shape [B]
        guide_fea : cuda Tensor, feature of one sample in the batch
            shape [B,C]

        Returns
        -------
        mean of gls_fea
            shape [B,C]
        '''
        mask = t.zeros(len_gls.shape[0], 1, max(len_gls)).bool().cuda()
        for i, l in enumerate(len_gls):
            mask[i, 0, l:] = True
        
        gls_fea = gls_fea.permute(2,0,1)  #[T,B,C]
        gls_fea += guide_fea
        gls_fea = gls_fea.transpose(0,1)  #[B,T,C]
        
        if self.seq_model_type == 'transformer':
            gls_fea, _ = self.seq_model(gls_fea, len_gls, mask)  #[B,T,C]
            gls_fea = MaskedMean(gls_fea.transpose(1,2), mask)  #sen-level-fea [B,C]
        elif self.seq_model_type == 'tcn':
            gls_fea, _, _ = self.seq_model(gls_fea.transpose(1,2), len_gls, mask) #[B,C,T]
            gls_fea = MaskedMean(gls_fea, mask)  #sen-level-fea [B,C]
        
        return gls_fea

























































