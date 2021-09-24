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
from modules.nets import resnet18_wo_fc, vgg11, mb_v2, googlenet
from modules.sgs import create_sgs_applier
from modules.tcn import TCN_block
from modules.cnn import CNN, pose_stream
from modules.dcn import DCN
from modules.aligner import Aligner
from utils.utils import create_mask, gen_random_mask, shuffle_tensor, MaskedMean

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
                 **kwargs
                 ):
        
        super(SLRModel, self).__init__()
        # self.p_detach = p_detach  #probability for stochatic gradient stop
        self.emb_size = args_model['emb_size']
        if args_model['name'] == 'lcsa':
            self.gloss_output_layer = nn.Linear(self.emb_size + pose_dim, gls_voc_size)
        elif args_model['name'] == 'fcn':
            self.gloss_output_layer = nn.Linear(self.emb_size*2, gls_voc_size)
        
        # visual logits
        self.ve = ve
        if ve:
            self.fc_for_ve = nn.Linear(self.emb_size, gls_voc_size)

        # semantics extractor
        self.sema_cons = sema_cons
        if sema_cons == 'drop':
            self.sema_ext = nn.Sequential(nn.Linear(self.emb_size, self.emb_size//4),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.emb_size//4, self.emb_size))
        elif sema_cons in ['shuffle', 'drop_shuffle']:
            self.sema_ext = RecurrentEncoder(rnn_type='gru',
                                            hidden_size=self.emb_size,
                                            emb_size=self.emb_size,
                                            num_layers=1,
                                            dropout=0.1,
                                            emb_dropout=0.1,
                                            bidirectional=False)
        
        self.vis_mod_type = args_model['vis_mod']
        self.seq_mod_type = args_model['seq_mod']
        
        # pretrained model paths
        path_dict = {'resnet18': '/2tssd/rzuo/pretrained_models/resnet18-5c106cde.pth',
                     'vgg11': '/2tssd/rzuo/pretrained_models/vgg11_bn-6002323d.pth',
                     'mb_v2': '/2tssd/rzuo/pretrained_models/mbv2.pth',
                     'mb_v2_ca': '/2tssd/rzuo/pretrained_models/mbv2_ca.pth',
                     'googlenet': '/2tssd/rzuo/pretrained_models/googlenet-1378be20.pth'}
        
        #visual module
        if self.vis_mod_type == 'resnet18':
            self.vis_mod = resnet18_wo_fc(spatial_att=spatial_att, pretrained=True, pre_model_path=path_dict['resnet18'])
        elif self.vis_mod_type == 'vgg11':
            self.vis_mod = vgg11(True, spatial_att, att_idx_lst, pool_type, cbam_no_channel, cbam_pool, freeze=False, pretrained=True, pre_model_path=path_dict['vgg11'])
        elif self.vis_mod_type == 'mb_v2':
            self.vis_mod = mb_v2(emb_size=self.emb_size, spatial_att=spatial_att, pretrained=True, pre_model_path=path_dict['mb_v2_ca'] if spatial_att=='ca' else path_dict['mb_v2'])
        elif self.vis_mod_type == 'googlenet':
            self.vis_mod = googlenet(pretrained=True, pre_model_path=path_dict['googlenet'])
        elif self.vis_mod_type == 'cnn':
            self.vis_mod = CNN(*pose_arg)
        elif self.vis_mod_type == 'dcn':
            self.vis_mod = DCN(dcn_ver, num_att=5)
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
                                                   use_pool=False),
                                         TCN_block(inchannels=self.emb_size,
                                                   outchannels=self.emb_size*2,
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
        elif self.seq_mod_type in ['lstm', 'gru']:
            self.seq_mod = RecurrentEncoder(rnn_type=self.seq_mod_type,
                                            hidden_size=self.emb_size,
                                            emb_size=self.emb_size,
                                            num_layers=1,
                                            dropout=0.1,
                                            emb_dropout=0.1,
                                            bidirectional=False)
        else:
            raise ValueError('We only support transformer and tcn now.\n')
            
        #pose stream
        self.pose_dim = pose_dim
        if pose_dim > 0:
            self.pose_mod = pose_stream()
        
    def forward(self, video, len_video, heatmap=None, return_att=False):
        assert video.shape[0] == t.sum(len_video).item()
        assert video.shape[1] == 3
        
        #visual module with stochstic gradient stopping
        # sgs_apply = create_sgs_applier(self.p_detach, len_video)
        offset, spat_mask, video = self.vis_mod(
            video, heatmap=None if self.pose_dim>0 else heatmap)  #[sum_T, 512]
        if self.pose_dim > 0:
            pose_fea = self.pose_mod(heatmap)
            video = t.cat([video, pose_fea], dim=1)
        video = video.split(len_video.tolist())
        
        mask = create_mask(len_video)  #True for padding
        video = pad_sequence(video, batch_first=True)  #[B, max_T, 512]
        
        semantics = vis_logits = None
        if self.ve:
            vis_logits = self.fc_for_ve(video)
        if self.sema_cons == 'drop':
            semantics = [self.sema_ext(video.mean(dim=1))]
        elif self.sema_cons == 'shuffle':
            semantics = [self.sema_ext(video, len_video)[1]]
        
        plot = None
        if self.seq_mod_type == 'transformer':
            video, plot = self.seq_mod(video, len_video, mask, return_att)  #[B,T,C]
        elif self.seq_mod_type in ['lstm', 'gru']:
            video, plot = self.seq_mod(video, len_video)
        elif self.seq_mod_type in ['tcn', 'tcntr']:
            video = video.transpose(1,2)
            for i in range(len(self.seq_mod)):
                if isinstance(self.seq_mod[i], TCN_block):
                    video, len_video, mask = self.seq_mod[i](video, len_video, mask)
                elif isinstance(self.seq_mod[i], TransformerEncoder):
                    video = video.transpose(1,2)
                    video, plot = self.seq_mod[i](video, len_video, mask, return_att)
            if self.seq_mod_type == 'tcn':
                video = video.transpose(1,2)
        
        if self.sema_cons == 'drop':
            # positive
            semantics.append(self.sema_ext(MaskedMean(video, mask, dim=1)))
            # negative using random mask
            sel_length, r_mask = gen_random_mask(len_video, drop_ratio=0.5)  #[B,T,1]
            fea_sel = video.masked_select(r_mask).view(-1, self.emb_size)  #[T1+T2,C]
            fea_sel = fea_sel.split(sel_length)
            fea_sel = pad_sequence(fea_sel).sum(dim=0)  #[B,C]
            sel_length = t.tensor(sel_length).unsqueeze(-1).cuda()  #[B,1]
            semantics.append(self.sema_ext(fea_sel/sel_length))
        elif self.sema_cons == 'shuffle':
            # positive
            semantics.append(self.sema_ext(video, len_video)[1])
            # negative using random shuffle
            semantics.append(self.sema_ext(shuffle_tensor(video, len_video), len_video)[1])
        elif self.sema_cons == 'drop_shuffle':
            # positive
            semantics.append(self.sema_ext(video, len_video)[1])
            # negative using drop and then random shuffle
            sel_length, r_mask = gen_random_mask(len_video, drop_ratio=0.5)  #[B,T,1]
            fea_sel = video.masked_select(r_mask).view(-1, self.emb_size)  #[T1+T2,C]
            fea_sel = fea_sel.split(sel_length)
            fea_sel = pad_sequence(fea_sel)
            semantics.append(self.sema_ext(fea_sel, t.tensor(sel_length))[1])

        video = self.gloss_output_layer(video)  #gls_scores
        return {'gls_logits': video,
                'vis_logits': vis_logits,
                'len_video': len_video,
                'dcn_offset': offset,
                'spat_att': spat_mask,
                'semantics': semantics,
                'plot': plot}

class CMA(nn.Module):
    def __init__(self, gls_voc_size=1233):
        super(CMA, self).__init__()
        self.googlenet = googlenet()
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


#*******************************SLP**********************************
class Simple_SLP(nn.Module):
    #A naive slp model
    def __init__(self, emb_size=512, voc_size=1233, 
                 gls_encoder_type='gru', img_encoder_type='cnn', 
                 pre_vision_model_path=None, use_aligner=True):
        super(Simple_SLP, self).__init__()
        self.padding_index = voc_size-2  #unk
        self.use_aligner = use_aligner
        self.emb = Embeddings(embedding_dim=emb_size,
                              num_heads=1,
                              norm_type='batch',
                              activation_type='relu',
                              vocab_size=voc_size,
                              padding_index=self.padding_index)
        
        self.gls_encoder = RecurrentEncoder(rnn_type=gls_encoder_type,
                                            hidden_size=emb_size,
                                            emb_size=emb_size,
                                            num_layers=1,
                                            dropout=0.1,
                                            emb_dropout=0.1,
                                            bidirectional=True)
        
        if img_encoder_type == 'cnn':
            self.img_encoder = CNN()
        elif img_encoder_type == 'resnet18':
            self.img_encoder = resnet18_wo_fc(pretrained=True, pre_model_path=pre_vision_model_path)
        
        if use_aligner:
            self.aligner = Aligner(emb_size=emb_size*2)
            
        self.seq_model = RecurrentEncoder(rnn_type=gls_encoder_type,
                                          hidden_size=emb_size,
                                          emb_size=emb_size*3,
                                          num_layers=2,
                                          dropout=0.1,
                                          emb_dropout=0.1,
                                          bidirectional=True)
        
        #image decoder(2D-deconv) or we can use video decoder(3D-deconv/2D-deconv + 1D-deconv) in the futu
        #[B,1024,T] -> [SUM,1024,1,1] -> [SUM,3,224,224]
        self.decoder = nn.Sequential([
            nn.ConvTranspose2d(1024, 512, kernel_size=7, bias=False),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh()
        ])
        
    
    def forward(self, gls, len_gls, guide_img, out_seq_len=299):
        '''
        Parameters
        ----------
        gls : LongTensor
            [SUM].
        len_gls : Tensor
            [B]
        guide_img : Tensor
            [B,3,224,224]
        out_seq_len : int
            max frame lenght in training dataset
        Returns
        -------
        gls : Tensor
            generated video, shape [B,max_T,3,224,224]
        len_gls : LongTensor
            length of generated video, shape [B]
        '''
        gls = gls.split(len_gls.tolist())
        gls = pad_sequence(gls, batch_first=True, padding_value=self.padding_index).long()  #[B,MAX_T,C]
        
        mask = t.zeros(len_gls.shape[0], 1, max(len_gls)).bool().cuda()
        for i, l in enumerate(len_gls):
            mask[i, 0, l:] = True
        
        #text/gls encoding
        gls = self.emb(gls, mask)  #[B,C,max_T], C=512
        B, C, T = gls.shape
        gls, _ = self.gls_encoder(gls.transpose(1,2), len_gls)  #[B,T,2C]
        
        #align
        if self.use_aligner:
            T = out_seq_len
            len_gls, gls = self.aligner(gls, len_gls, out_seq_len=out_seq_len)  #[B,seq,2C]
            len_gls = t.min(len_gls, t.Tensor([out_seq_len]*B))
            mask = t.zeros(len_gls.shape[0], 1, out_seq_len).bool().cuda()
            for i, l in enumerate(len_gls):
                mask[i, 0, l:] = True
        
        #concatenate with guide feature
        guide_img = self.img_encoder(guide_img)  #[B,C]
        guide_img = guide_img.expand(T,B,C).transpose(0,1)  #[B,T,C]
        gls = t.cat([gls, guide_img], dim=-1)  #[B,T,3C]
        
        #sequence modeling
        gls, _ = self.seq_model(gls.transpose(1,2), len_gls)  #[B,seq,2C]
        
        #image/video decoding
        gls = gls.reshape(-1, C*2)  #[SUM,2C]
        mask = mask.reshape(-1, 1) <= 0
        gls = t.masked_select(gls, mask).reshape(-1, C*2, 1, 1)  #[real_sum,1024,1,1]
        gls = self.decoder(gls)  #[real_sum,3,224,224]
        gls = gls.split(len_gls.tolist())
        gls = pad_sequence(gls, batch_firs=True)  #[B,max_T,3,224,224]
        
        return gls, len_gls

# if __name__ == '__main__':
    # video = t.rand(15,3,224,224).cuda()
    # len_video = t.LongTensor([3,5,7]).cuda()
    # model = SetrModel(pre_vision_model_path='../../pretrained_models/resnet18-5c106cde.pth').cuda()
    # device = next(model.parameters()).device
    # print(device)
    # out = model(video, len_video)
    # print(out.shape)