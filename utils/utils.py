# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:20:25 2020

@author: 14048
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import random
from itertools import groupby
import numpy
# from ptflops import get_model_complexity_info
from mmcv import runner
from collections import defaultdict


def init_logging(log_file):
    """Init for logging
    """
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s: %(message)s',
                        datefmt = '%m-%d %H:%M:%S',
                        filename = log_file,
                        filemode = 'a+')
    # define a Handler which writes INFO message or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class LossManager(object):
    def __init__(self, print_step):
        self.print_step = print_step
        self.last_state = {}
        self.record = defaultdict(list)
    
    def update(self, loss_dict, global_step):
        for key in loss_dict.keys():
            self.record[key].append(loss_dict[key])
        if (global_step % self.print_step) == 0:
            for key in self.record.keys():
                self.record[key] = np.mean(self.record[key])
            if not self.last_state:
                # last_state is empty
                self.last_state = self.record.copy()
            info = 'Global step {:d}'.format(global_step)
            for key in self.record.keys():
                info += ', {:s}: {:.5f} -> {:.5f}'.format(key, self.last_state[key], self.record[key])
            self.last_state = self.record.copy()
            self.record = defaultdict(list)
            logging.info(info)


def record_loss(loss_dict, record_dict):
    for key in loss_dict.keys():
        record_dict[key].append(loss_dict[key])


class ModelManager(object):
    def __init__(self, max_num_models=5):
        self.max_num_models = max_num_models
        self.best_epoch = 0
        self.best_err = np.ones([4])*1000
        self.model_file_list = []

    def update(self, model_file, err, epoch, lr_scheduler, lr_scheduler_D=None, scheduler_type='plateau'):
        self.model_file_list.append((model_file, err))
        self.update_best_err(err, epoch)
        self.sort_model_list()
        if len(self.model_file_list) > self.max_num_models:
            worst_model_file = self.model_file_list.pop(-1)[0]
            if os.path.exists(worst_model_file):
                os.remove(worst_model_file)
        
        #update lr
        if scheduler_type == 'plateau':
            lr_scheduler.step(err[0])
            if lr_scheduler_D is not None:
                lr_scheduler_D.step(err[0])
        elif 'step' in scheduler_type:
            lr_scheduler.step()
            
        now_lr = lr_scheduler.optimizer.param_groups[0]["lr"]
        logging.info('CURRENT BEST PERFORMANCE (epoch: {:d}): WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}. Now lr: {:.8f}'.format( \
            self.best_epoch, self.best_err[0], self.best_err[1], self.best_err[2], self.best_err[3], now_lr))
        pass

    def update_best_err(self, err, epoch):
        if err[0] < self.best_err[0]:
            self.best_err = err
            self.best_epoch = epoch

    def sort_model_list(self):
        self.model_file_list.sort(key=lambda x: x[1][0])


def setup_seed(seed=8):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    runner.set_random_seed(seed, deterministic=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

def worker_init_fn(worker_id):         
    # https://pytorch.org/docs/stable/notes/randomness.html              
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def create_mask(len_video):
    b_size = len_video.shape[0]
    mask = torch.zeros(b_size, 1, max(len_video)).bool().cuda()
    for i, l in enumerate(len_video):
        mask[i, 0, l:] = True
    return mask  #[B,1,T]


def gen_random_mask(len_video, drop_ratio=0.5):
    b_size = len_video.shape[0]
    keep_ratio = 1-drop_ratio
    mask = torch.zeros(b_size, max(len_video), 1).bool().cuda()
    sel_length = []
    for i, l in enumerate(len_video):
        l = l.item()
        length = int(l*keep_ratio)
        sel_length.append(length)
        idx = np.arange(l)
        np.random.shuffle(idx)
        idx = idx[:length]
        mask[i, idx, 0] = True
    return sel_length, mask  #[B,T,1]


def shuffle_tensor(video, len_video):
    # shuffle a video tensor [B,T,C]
    l = max(len_video)
    shuffled_idx = torch.randperm(l).cuda()
    return video.index_select(1, shuffled_idx)


def gen_neg_sample(video, way, drop_ratio=0.5):
    T = video.shape[1]
    if way == 'shuffle':
        idx = torch.randperm(T).cuda()
    elif way in ['multi_shuffle', 'batch_shuffle']:
        # shuffle for many times until more than 50% idx are differnet
        label = torch.arange(T)
        idx = torch.randperm(T)
        while((idx==label).sum().item() > (1-drop_ratio)*T):
            idx = torch.randperm(T)
        idx = idx.cuda()
        if way == 'batch_shuffle' and video.shape[0] == 2:
            video = video.index_select(0, torch.tensor([1,0]).cuda())
    elif way == 'drop':
        idx = torch.arange(T).cuda()
        idx = idx[:int(T*(1-drop_ratio))]
    elif way == 'drop_shuffle':
        idx = torch.randperm(T).cuda()
        idx = idx[:int(T*(1-drop_ratio))]
    elif way == 'drop_shuffle_insert':
        idx = torch.randperm(T).cuda()
        idx = idx[:int(T*(1-drop_ratio))]
        i = torch.randint(high=idx.shape[0], size=(T,))
        idx = idx[i]
    return video.index_select(1, idx)


def MaskedMean(x, len_video, dim=-1):
    '''
    Parameters
    ----------
    x : Tensor
        shape [B,C,max_T] or [B,max_T,C]
    mask : bool Tensor
        shape [B,1,max_T]. True for padding values!
    dim : dimension
        The default is -1.

    Returns
    -------
    mean
        shape [B,C]
    '''
    mask = create_mask(len_video)
    x = x.sum(dim)
    x = x.transpose(0,1)  #[C,B]
    mask = (mask<=0).squeeze(1).sum(-1)
    x /= mask
    return x.transpose(0,1)  #[B,C]


def update_dict(state_dict):
    new_state_dict = state_dict.copy()
    for key in state_dict.keys():
        if 'encoder' in key:
            new_key = key.replace('encoder', 'seq_mod')
            new_state_dict[new_key] = state_dict[key]
            new_state_dict.pop(key)
        elif 'vi_fea_ext' in key:
            new_key = key.replace('vi_fea_ext', 'vis_mod')
            new_state_dict[new_key] = state_dict[key]
            new_state_dict.pop(key)
        elif 'vis_mod.head' in key:
            print(key, new_state_dict[key].shape)
            new_state_dict.pop(key)
    
    return new_state_dict


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False
        
        
def unfreeze_params(module: nn.Module):
    """
    Unfreeze the parameters of this module,
    i.e. do not update them during training

    :param module: unfreeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = True
        

def get_param_count(model):
    return sum(torch.numel(p) for p in model.parameters() if p.requires_grad) / 1e6


# def get_flop_count(model):
#     #inputs should be a tuple
#     flops, _ = get_model_complexity_info(model, (3,224,224), as_strings=True, 
#                                          print_per_layer_stat=True, verbose=True)
#     return flops
    