# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:02:21 2021

@author: Ronglai ZUO
figure-related
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import os; os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch as t
# import seaborn as sns; sns.set_theme()


def gen_att_map(data, save_file):
    #generate attention map
    #data tensor, shape [T,T]
    data = data.cpu().numpy()
    plt.figure()
    # sns.heatmap(data, vmin=0, vmax=1, cmap='YlGnBu')
    plt.savefig(save_file)


def gate_wer_fig(dir_lst):
    data = []
    for dir in dir_lst:
        fname = dir+'/D_wer.npz'
        dic = np.load(fname)
        wer = dic['wer']
        gate = dic['D']
        mean_gate = np.mean(gate)
        print(mean_gate, dir.split('/')[-1])
        # print(wer.shape, gate.shape)
        # idx = np.argsort(wer)
        # wer = wer[idx]
        # gate = gate[idx]
        data.append([wer, gate, mean_gate*np.ones(wer.shape[0])])
    
    x = np.arange(data[0][0].shape[0])
    # print(data[0][0])
    plt.figure()
    plt.plot(x, data[0][1], label='magnitudes of D')
    # plt.plot(x, data[1][1], label='RPE+Gau+F')
    plt.plot(x, data[0][2], '--', label='mean of D')
    # plt.plot(x, data[1][2], '--', label='RPE+Gau+F_mean')
    plt.xlabel('sample')
    plt.ylabel('magnitudes of D')
    plt.ylim(12.59,12.601)
    plt.legend()
    # y_major_locator = MultipleLocator(0.005)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.savefig('/2tssd/rzuo/codes/lcasan/utils/D_wer.jpg')
    

def wer_ratio_fig(dir):
    fname = dir+'/wer_ratio.npz'
    dic = np.load(fname)
    wer = dic['wer']
    ratio = dic['ratio']
    
    sort_idx = np.argsort(ratio)
    wer = wer[sort_idx]
    ratio = ratio[sort_idx]
    
    plt.figure()
    plt.plot(ratio, wer)
    plt.xlabel('ratio')
    plt.ylabel('wer')
    plt.savefig('/2tssd/rzuo/codes/setr/utils/wer_ratio.pdf')


def D_wer_fig(dir_spec, dir_share):
    fname = dir_spec + '/D_wer.npz'
    dic = np.load(fname)
    D1 = dic['D_1']
    D2 = dic['D_2']
    x = np.arange(D1.shape[0])
    
    f_share = dir_share + '/D_wer.npz'
    dic = np.load(f_share)
    D1_share = dic['D_1']
    D2_share = dic['D_2']
    m1_share = np.mean(D1_share)
    m2_share = np.mean(D2_share)
    
    plt.figure()
    # for i in range(D.shape[1]):
    #     # plot for each head
    #     plt.subplot(2,4,i+1)
    #     mean_D = np.mean(D[:, i])
    #     print(mean_D, dir.split('/')[-1])
    #     mean_D = mean_D * np.ones(D.shape[0])
    #     plt.plot(x, D[:, i], label='magnitudes of D')
    #     plt.plot(x, mean_D, '--', label='mean of D')
    #     plt.xlabel('sample')
    #     plt.ylabel('magnitudes of D')
    #     plt.legend()
    # for i in range(D.shape[1]):
    #     mean_D = np.mean(D[:, i])
    #     print(mean_D, dir.split('/')[-1])
    #     mean_D = mean_D * np.ones(D.shape[0])
    #     plt.plot(x, mean_D, label='head_{:d}'.format(i+1))
    mean_D1, mean_D2 = np.mean(D1, axis=0), np.mean(D2, axis=0)
    m1, m2 = np.mean(mean_D1), np.mean(mean_D2)
    print(mean_D1, m1, mean_D2, m2)
    x = np.arange(D1.shape[1])
    bar1 = plt.bar(x-0.2, mean_D1, width=0.4, label='spec_1st')
    bar2 = plt.bar(x+0.2, mean_D2, width=0.4, label='spec_2nd')
    for rect in bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, "%.1f"%height, ha="center", va="bottom", fontsize=10)
    for rect in bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, "%.1f"%height, ha="center", va="bottom", fontsize=10)
    plt.axhline(m1_share, ls='dotted', color='green', label='share_1st')
    plt.axhline(m2_share, ls='--', color='green', label='share_2nd')
    plt.xlabel('head index')
    plt.ylabel('window size')
    plt.legend()
    plt.savefig('/2tssd/rzuo/codes/lcasan/utils/D.pdf')


def hist(fname):
    dic = np.load(fname)
    n = fname.split('_')[-1]
    num_frame = dic['frame']
    print(max(num_frame), np.sum(num_frame>150))
    num_gls = dic['gls']
    ratio = num_frame / num_gls
    plt.hist(ratio, bins=20, density=True)
    # plt.hist(num_frame, bins=20)
    # plt.savefig('/2tssd/rzuo/codes/lcasan/utils/frame_len_' + n +'.jpg')
    
    x = np.arange(min(ratio), max(ratio), 0.1)
    mu = np.mean(ratio)
    sigma = np.std(ratio, ddof=1)
    y = 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/2/sigma**2)
    plt.plot(x, y, color='red')
    plt.xlabel('frame length / gloss length')
    plt.ylabel('frequency')
    plt.savefig('/2tssd/rzuo/codes/lcasan/utils/ratio_' + n + '.pdf')
    


# if __name__ == '__main__':
#     x = np.arange(4, 9)
#     y1 = np.array([23.1, 23.1 ,23.6, 23.4, 22.7])
#     y2 = np.array([22.4, 22.8 ,22.8, 22.7, 23.0])
#     plt.figure()
#     plt.plot(x, y1, color='blue', marker='v', label='RPE+Gau')
#     plt.plot(x, y2, '--', color='red', marker='v', label='RPE+Gau+F(cascade)')
#     plt.xlabel('D')
#     plt.ylabel('WER%')
#     #plt.ylim(22.8, 24.0)
#     plt.legend()
#     plt.savefig('D:\\HKUST\\research\\codes\\setr\\utils\\D.pdf')
    
    # wer_ratio_fig('/2tssd/rzuo/codes/setr/results/cnntr2_gau_lea_D10.0_conv_cas_nobias')


if __name__ == '__main__':
    # dir_lst = ['/2tssd/rzuo/codes/lcasan/results/lcasan_rpe_gau_modDfromQ_div2_qk10_cas']
              #'/2tssd/rzuo/codes/setr/results/cnntr2_gau_lea_D4.0_conv_gate']
    # gate_wer_fig(dir_lst)
    # dir_spec = '/2tssd/rzuo/codes/lcasan/results/lcasan_vgg11_rpe_gau_head_spec_qk10_2std'
    # dir_share = '/2tssd/rzuo/codes/lcasan/results/lcasan_vgg11_rpe_gau_D6.3_head_share_qk10_div2_2std'
    # dir_spec = '/2tssd/rzuo/codes/lcasan/results/lcasan_vgg11_rpe_gau_head_spec_2std'
    # dir_share = '/2tssd/rzuo/codes/lcasan/results/lcasan_vgg11_rpe_gau_head_share'
    # D_wer_fig(dir_spec, dir_share)
    hist('/2tssd/rzuo/codes/lcasan/phoenix_datasets/stat_csl1_seganddrop.npz')
