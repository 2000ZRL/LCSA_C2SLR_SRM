# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:06:36 2021

@author: 14048
"""
from time import sleep
import os

def main():
    sleep(5*3600)
    # os.system("python main.py --comb_conv='cas_bef_san' --ve=1 --va=1 --setting='semi_100' --save_dir='./results/vgg11_cas_bef_san_ve_dcs2v' --gpu=1")
    # os.system("python main.py --comb_conv=cas_bef_san --pl=1 --setting=semi_10 --num_pretrain_epoch=15 --save_dir='./results/vgg11_cas_bef_san_semi_10_pl' --gpu=0")
    # os.system("python main.py --va=1 --ve=1 --alpha=10.0 --comb_conv='cas_bef_san' --save_dir='./results/vgg11_cbs_vac_alpha10' --gpu=1 --from_ckpt=1")
    # os.system("python main.py --sema_cons=cosine --lr_factor=0.1 --comb_conv='cas_bef_san' --save_dir='./results/vgg11_cbs_sc_tfmer_ape_dtcn_cosine' --gpu=0")
    # os.system("python main.py --config=config/csl1.yml --comb_conv=cas_bef_san --spatial_att=cbam --cbam_no_channel=1 --cbam_pool=max_softmax --att_idx_lst 4 --pose=super_att --sema_cons=batch --lr_factor=0.1 --save_dir=results/COR_CSL1_60ep_deconemore_nosplitsen_vgg11_cbs_C2SLR --gpu=0")
    # os.system("python main.py --config=config/2014SI.yml --fde=distill_share --att_idx_lst 4 --fde_loss_w 1 1 0 0.001 --pose=prior --save_dir=results/SI_fde_pose_cam15_jsdshare_mutual --gpu=1")
    # os.system("python main.py --config=config/csl1.yml --fde=xvec_rev --att_idx_lst 4 --pose=super_att --sema_cons=batch --fde_loss_w 0.75 1 0 0 --lr_factor=0.1 --save_dir=results/COR_CSL1_60ep_deconemore_nosplitsen_C2SLR_presigner_gradrev_jmlr.75 --gpu=0")
    
    # os.system("python gen_heatmaps.py --dataset=csl-daily --split=train --gpu=2")
    # os.system("python gen_heatmaps.py --dataset=2014T --split=train --gpu=2")
    # os.system("python gen_flow.py --dataset=2014T --split=dev --max_num_frame=100 --gpu=2")
    # os.system("python gen_flow.py --dataset=2014T --split=test --max_num_frame=100 --gpu=2")
    # os.system("python gen_flow.py --dataset=csl-daily --split=dev --max_num_frame=100 --gpu=2")
    # os.system("python gen_flow.py --dataset=csl-daily --split=test --max_num_frame=100 --gpu=2")
if __name__ == '__main__':
    main()