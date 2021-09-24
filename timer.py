# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:06:36 2021

@author: 14048
"""
from time import sleep
import os

def main():
    sleep(4*3600)
    # os.system("python main.py --vis_mod='vgg11' --qkv_context 1 0 0 --mod_D='head_specific' --spatial_att='cbam' --cbam_pool='max_softmax' --att_idx_lst 5 6 7 --pose='super_att' --att_sup_type='first' --heatmap_shape 28 --save_dir='./results/lcasan_vgg11_rpe_gau_qk10_head_spec_cbam_last3_max_softmax_superatt_1st' --gpu=0 --max_num_epoch=50 --patience=6")
    # os.system("python main.py --vis_mod='vgg11' --spatial_att='dcn' --att_idx_lst 5 6 7 --pose='super_att' --att_sup_type='first' --comb_conv='cas_bef_san' --save_dir='./results/lcasan_vgg11_dcn_nooff_last3_superatt' --gpu=0 --max_num_epoch=60 --heatmap_shape 28")
    # os.system("python main.py --comb_conv='cas_bef_san' --ve=1 --va=1 --setting='semi_100' --save_dir='./results/vgg11_cas_bef_san_ve_dcs2v' --gpu=1")
    os.system("python main.py --comb_conv=cas_bef_san --pl=1 --setting=semi_10 --num_pretrain_epoch=15 --save_dir='./results/vgg11_cas_bef_san_semi_10_pl' --gpu=0")
if __name__ == '__main__':
    main()