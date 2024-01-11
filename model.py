# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:25:56 2020

@author: Ronglai ZUO
"""

import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tfmer.encoders import TransformerEncoder, RecurrentEncoder
from tfmer.embeddings import Embeddings
from modules.nets import resnet18_wo_fc, vgg11, cnn9, mb_v2, googlenet#, MobileNet_v3
from modules.tcn import TCN_block
from utils.utils import gen_random_mask, MaskedMean, gen_neg_sample

class SLRModel(nn.Module):
    def __init__(self, 
                 args_model,
                 args_tf,
                 D_std_gamma=[6.3,1.4,2.0],
                 mod_D=None,
                 mod_src='Q',
                 comb_conv=None,
                 qkv_context=[0,0,0],
                 gls_voc_size=1233, 
                 pose_arg=['filter',3,0.5],
                 pose_dim=0,
                 dcn_ver='v2',
                 att_idx_lst=[],
                 spatial_att=None,
                 pool_type='avg',
                 cbam_no_channel=False,
                 cbam_pool='max_avg',
                 ve=False,
                 sema_cons=None,
                 drop_ratio=0.5,
                 **kwargs
                 ):
        
        super(SLRModel, self).__init__()
        # self.p_detach = p_detach  #probability for stochatic gradient stop
        self.emb_size = args_model['emb_size']
        if args_model['name'] == 'lcsa':
            self.gloss_output_layer = nn.Linear(self.emb_size + pose_dim, gls_voc_size)
        else:
            self.gloss_output_layer = nn.Linear(self.emb_size*2, gls_voc_size)
        
        # visual logits
        self.fde = kwargs.pop('fde', None)
        self.ve = ve
        if ve or self.fde == 'distill':
            self.fc_for_ve = nn.Linear(self.emb_size, gls_voc_size)

        # semantics extractor
        self.sema_cons = sema_cons
        if sema_cons == 'mask':
            self.sema_ext = nn.Sequential(nn.Linear(self.emb_size, self.emb_size//4),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.emb_size//4, self.emb_size))
        elif sema_cons is not None:            
            args_sema_tf={'tf_model_size': 512, 'tf_ff_size': 2048, 'num_layers': 1, 'num_heads': 8, 'dropout': 0.1, 'emb_dropout': 0.1, 'pe': 'ape'}
            self.sema_ext =  TransformerEncoder(args_sema_tf,
                                                comb_conv='cas_bef_san',
                                                need_cls_token='frame' if sema_cons=='frame' else 'sen',
                                                freeze=False)

        self.drop_ratio = drop_ratio
        
        self.vis_mod_type = args_model['vis_mod']
        self.seq_mod_type = args_model['seq_mod']
        
        # pretrained model paths
        path_dict = {'resnet18': '../../pretrained_models/resnet18-5c106cde.pth',
                     'vgg11': '../../pretrained_models/vgg11_bn-6002323d.pth',
                     'mb_v2': '../../pretrained_models/mbv2.pth',
                     'mb_v2_ca': '../../pretrained_models/mbv2_ca.pth',
                     'googlenet': '../../pretrained_models/googlenet-1378be20.pth',
                     'deit_tiny': '../../pretrained_models/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
                     'deit_small': '../../pretrained_models/deit_small_distilled_patch16_224-649709d9.pth',
                     'deit_base': '../../pretrained_models/deit_base_distilled_patch16_224-df68dfff.pth',
                     'deit_small_nodist': '../../pretrained_models/deit_small_patch16_224-cd65a155.pth'
                     }
        
        #visual module
        if self.vis_mod_type == 'resnet18':
            self.vis_mod = resnet18_wo_fc(spatial_att=spatial_att, cbam_no_channel=cbam_no_channel, cbam_pool=cbam_pool, pretrained=True, pre_model_path=path_dict['resnet18'])
        elif self.vis_mod_type == 'vgg11':
            self.vis_mod = vgg11(True, spatial_att, att_idx_lst, pool_type, cbam_no_channel, cbam_pool, 
                                freeze=False, pretrained=True, pre_model_path=path_dict['vgg11'], 
                                fde=self.fde, num_signers=kwargs.pop('num_signers', 8))
        elif self.vis_mod_type == 'mb_v2':
            self.vis_mod = mb_v2(emb_size=self.emb_size, spatial_att=spatial_att, pretrained=True, pre_model_path=path_dict['mb_v2_ca'] if spatial_att=='ca' else path_dict['mb_v2'])
        # elif self.vis_mod_type == 'mb_v3_large':
        #     self.vis_mod = MobileNet_v3('large', spatial_att=spatial_att)
        # elif self.vis_mod_type == 'mb_v3_small':
        #     self.vis_mod = MobileNet_v3('small', spatial_att=spatial_att)
        elif self.vis_mod_type == 'googlenet':
            self.vis_mod = googlenet(pretrained=True, pre_model_path=path_dict['googlenet'])
        elif self.vis_mod_type == 'cnn':
            self.vis_mod = cnn9(True, spatial_att, att_idx_lst, pool_type, cbam_no_channel, cbam_pool)
        # elif self.vis_mod_type == 'dcn':
        #     self.vis_mod = DCN(dcn_ver, num_att=5)
        # elif self.vis_mod_type == 'deit_small':
        #     self.vis_mod = deit_small_distilled_patch16_224(pretrained=True, pre_model_path=path_dict['deit_small'])
        else:
            raise ValueError('We only support resnet18, CNN and DCN now.\n')
        
        #sequential module
        if self.seq_mod_type == 'transformer':
            self.seq_mod = TransformerEncoder(args_tf,
                                              D_std_gamma=D_std_gamma,
                                              mod_D=mod_D,
                                              mod_src=mod_src,
                                              comb_conv=comb_conv,
                                              qkv_context=qkv_context,
                                              freeze=False)
        elif self.seq_mod_type == 'tcn':
            self.seq_mod = nn.ModuleList([TCN_block(inchannels=self.emb_size,
                                                   outchannels=self.emb_size,
                                                   kernel_size=5,
                                                   stride=1,
                                                   use_pool=False),
                                         TCN_block(inchannels=self.emb_size,
                                                   outchannels=self.emb_size,
                                                   kernel_size=5,
                                                   stride=1,
                                                   use_pool=False),  #True
                                         TCN_block(inchannels=self.emb_size,
                                                   outchannels=self.emb_size,  #emb_size*2
                                                   kernel_size=3,
                                                   stride=1,
                                                   use_pool=False)
                                         ])
        elif self.seq_mod_type == 'tcntr':
            self.seq_mod = nn.ModuleList([TCN_block(inchannels=self.emb_size,
                                                   outchannels=self.emb_size,
                                                   kernel_size=5,
                                                   stride=1,
                                                   groups=self.emb_size,
                                                   use_pool=False),
                                         TCN_block(inchannels=self.emb_size,
                                                   outchannels=self.emb_size,
                                                   kernel_size=5,
                                                   stride=1,
                                                   groups=self.emb_size,
                                                   use_pool=False),
                                         TransformerEncoder(args_tf,
                                                          D_std_gamma=D_std_gamma,
                                                          mod_D=mod_D,
                                                          mod_src=mod_src,
                                                          comb_conv=comb_conv,
                                                          qkv_context=qkv_context,
                                                          freeze=False)])
        elif self.seq_mod_type == 'tcnbilstm':
            self.seq_mod = nn.ModuleList([TCN_block(inchannels=self.emb_size,
                                                   outchannels=self.emb_size,  #embsize*2
                                                   kernel_size=5,
                                                   stride=1,
                                                   use_pool=False),
                                        TCN_block(inchannels=self.emb_size,
                                                   outchannels=self.emb_size,
                                                   kernel_size=5,
                                                   stride=1,
                                                   use_pool=False),  #True
                                        RecurrentEncoder(rnn_type='lstm',
                                            hidden_size=self.emb_size//2,  #emb_size
                                            emb_size=self.emb_size,
                                            num_layers=2,
                                            dropout=0.1,
                                            emb_dropout=0.1,
                                            bidirectional=True)])
        elif self.seq_mod_type in ['lstm', 'gru']:
            self.seq_mod = nn.ModuleList([RecurrentEncoder(rnn_type=self.seq_mod_type,
                                            hidden_size=self.emb_size,
                                            emb_size=self.emb_size,
                                            num_layers=1,
                                            dropout=0.1,
                                            emb_dropout=0.1,
                                            bidirectional=True),
                                        RecurrentEncoder(rnn_type=self.seq_mod_type,
                                            hidden_size=self.emb_size,
                                            emb_size=self.emb_size*2,
                                            num_layers=1,
                                            dropout=0.1,
                                            emb_dropout=0.1,
                                            bidirectional=True)
            ])
        else:
            raise ValueError('We only support transformer and tcn now.\n')
            
        #pose stream
        # self.pose_dim = pose_dim
        # if pose_dim > 0:
        #     self.pose_mod = pose_stream()
        
    def forward(self, video, len_video, **kwargs):
        coord = kwargs.pop('coord', None); return_att = kwargs.pop('return_att', False)
        signer = kwargs.pop('signer', None); signer_emb_bank = kwargs.pop('signer_emb_bank', {})
        assert video.shape[0] == t.sum(len_video).item()
        assert video.shape[1] == 3
        
        #visual module with stochstic gradient stopping
        # sgs_apply = create_sgs_applier(self.p_detach, len_video)
        vis_dict = self.vis_mod(video, coord=coord, signer=signer, signer_emb_bank=signer_emb_bank, len_video=len_video)
        offset, spat_mask, video = vis_dict['offset_lst'], vis_dict['mask_lst'], vis_dict['output']  #[sum_T, 512]
        video = video.split(len_video.tolist())
        
        # mask = create_mask(len_video)  #True for padding
        video = pad_sequence(video, batch_first=True)  #[B, max_T, 512]
        
        semantics = vis_logits = None
        if self.ve and self.training:
            vis_logits = self.fc_for_ve(video)
        
        if self.sema_cons == 'mask':
            semantics = [self.sema_ext(video.mean(dim=1))]
        elif self.sema_cons == 'frame':
            semantics = [self.sema_ext(video, len_video)[1].transpose(1,2)]
        elif self.sema_cons is not None:
            semantics = [self.sema_ext(video, len_video)[1]]
        
        # if self.sema_cons is not None and 'visual' in self.sema_cons:
        #     # for this situation, anc=sequential feature, pos=visual feature, neg=negative visual feature
        #     drop_ratio = 0.5
        #     semantics.append(self.sema_ext(gen_neg_sample(video, 'drop_shuffle', drop_ratio), (1-drop_ratio)*len_video.long())[1])

        plot = None
        if self.seq_mod_type == 'transformer':
            video, plot = self.seq_mod(video, len_video, return_att)  #[B,T,C]
        elif self.seq_mod_type in ['lstm', 'gru']:
            for i in range(len(self.seq_mod)):
                video, plot = self.seq_mod[i](video, len_video)
        elif self.seq_mod_type in ['tcn', 'tcntr', 'tcnbilstm']:
            video = video.transpose(1,2)
            for i in range(len(self.seq_mod)):
                if isinstance(self.seq_mod[i], TCN_block):
                    video, len_video = self.seq_mod[i](video, len_video)
                elif isinstance(self.seq_mod[i], TransformerEncoder):
                    video = video.transpose(1,2)
                    video, plot = self.seq_mod[i](video, len_video, return_att)
                elif isinstance(self.seq_mod[i], RecurrentEncoder):
                    video = video.transpose(1,2)
                    video, _ = self.seq_mod[i](video, len_video)
            if self.seq_mod_type == 'tcn':
                video = video.transpose(1,2)
        
        if self.sema_cons == 'mask':
            # positive
            semantics.append(self.sema_ext(MaskedMean(video, len_video, dim=1)))
            # negative using random mask
            sel_length, r_mask = gen_random_mask(len_video, drop_ratio=0.5)  #[B,T,1]
            fea_sel = video.masked_select(r_mask).view(-1, self.emb_size)  #[T1+T2,C]
            fea_sel = fea_sel.split(sel_length)
            fea_sel = pad_sequence(fea_sel, batch_first=True)  #[B,T,C]
            sel_length = t.tensor(sel_length).unsqueeze(-1).cuda()  #[B,1]
            semantics.append(self.sema_ext(fea_sel.sum(dim=1)/sel_length))
        
        elif self.sema_cons == 'frame':
            semantics.append(self.sema_ext(video, len_video)[1].transpose(1,2))
        
        elif self.sema_cons in ['batch', 'sequential', 'cosine']:
            # positive
            semantics.append(self.sema_ext(video, len_video)[1])

        elif self.sema_cons is not None:
            # positive
            semantics.append(self.sema_ext(video, len_video)[1])
            # negative
            if 'drop' in self.sema_cons and 'insert' not in self.sema_cons:
                len_neg_sample = ((1-self.drop_ratio)*len_video).long()
            else:
                len_neg_sample = len_video
            semantics.append(self.sema_ext(gen_neg_sample(video, self.sema_cons, self.drop_ratio), len_neg_sample)[1])

        video = self.gloss_output_layer(video)  #gls_scores
        if self.fde in ['distill', 'distill_share'] and self.training:
            vis_logits = vis_dict['signer_emb']
            vis_logits = vis_logits.split(len_video.tolist())
            vis_logits = pad_sequence(vis_logits, batch_first=True)
            if self.fde == 'distill':
                vis_logits = self.fc_for_ve(vis_logits)
            else:
                vis_logits = self.gloss_output_layer(vis_logits)
        
        return {'gls_logits': video,
                'vis_logits': vis_logits,
                'len_video': len_video,
                'offset': offset,
                'spat_att': spat_mask,
                'semantics': semantics,
                'cam': vis_dict.pop('cam', None),
                'ch_cam': vis_dict.pop('ch_cam', None),
                'signer_emb': vis_dict.pop('signer_emb', None),
                'signer_logits': vis_dict.pop('signer_logits', None),
                'plot': plot}

class CMA(nn.Module):
    def __init__(self, gls_voc_size=1233):
        super(CMA, self).__init__()
        self.googlenet = googlenet(pretrained=False)
        self.tcn_block = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
                                       nn.MaxPool1d(kernel_size=2),
                                       nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
                                       nn.MaxPool1d(kernel_size=2))
        self.blstm = RecurrentEncoder(rnn_type='lstm',
                                      hidden_size=512,
                                      emb_size=1024,
                                      num_layers=2,
                                      dropout=0,
                                      emb_dropout=0,
                                      bidirectional=True)
        self.gloss_output_layer = nn.Linear(1024, gls_voc_size)
    
    def forward(self, video, len_video):
        video = self.googlenet(video)
        video = video.split(len_video.tolist())
        video = pad_sequence(video, batch_first=True)  #[B,T,1024]
        video = self.tcn_block(video.transpose(1,2))  #[B,1024,T]
        len_video = len_video//4
        video, _ = self.blstm(video.transpose(1,2), len_video)  #[B,T,1024]
        video = self.gloss_output_layer(video)
        
        return video, len_video, None, None

