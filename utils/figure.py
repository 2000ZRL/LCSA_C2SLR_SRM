# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:02:21 2021

@author: Ronglai ZUO
figure-related
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator

import os; os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch as t
import pandas as pd
import seaborn as sns



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


def sc_box(dir_lst):
    # positive and negative distance of sentence embedding consistency
    diff_lst = []
    for dir in dir_lst:
        data = np.load(dir+'/sc_dist.npz')
        data = data['dist']
        assert data.shape[1] == 4
        y_pos = data[:, :2].mean(axis=-1)
        y_neg = data[:, 2:].mean(axis=-1)
        # y_pos = y_pos[:1400]
        # y_neg = y_neg[:1400]
        diff = y_pos - y_neg
        diff_lst.append(diff)

    # x = np.arange(0, max(y_neg), 0.01)
    # y = x.copy()
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.scatter(y_neg, y_pos, marker='o', color='b', s=5, alpha=0.15)
    # # ax.scatter(x, y_neg, marker='o', color='g', label='negative')
    # ax.plot(x, y, ls='--', color='r', lw=1.0)
    # ax.set_xlabel('negative distance')
    # ax.set_ylabel('positive distance')
    # # ax.legend()
    # plt.savefig(dir+'/sc_dist.jpg')

    bp = ax.boxplot(diff_lst, vert=False, labels=['sentence', 'frame'], widths=[0.8,0.8], patch_artist=True, showmeans=True, meanline=True)
    plt.setp(bp['medians'], color='blue')
    for patch in bp['boxes']:
        patch.set(facecolor='pink')
    plt.xlim(-2, 2)
    plt.ylim(0.5, 8)
    # plt.yticks([1,5])
    plt.savefig(dir_lst[0]+'/sc_dist_box.pdf')


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


def hist_cw(dir):
    # hist for channel weights in max_softmax CBAM
    data1 = np.load(dir+'/cw_test0.npz')
    data1 = data1['cw']
    data2 = np.load(dir+'/cw_test10.npz')
    data2 = data2['cw']
    print(data1.shape, data2.shape)
    data1, data2 = data1.mean(axis=0), data2.mean(axis=0)
    # print(data1.sum())

    # estimate gaussian
    mu1, sigma1 = np.mean(data1), np.std(data1, ddof=1)
    mu2, sigma2 = np.mean(data2), np.std(data2, ddof=1)
    x1, x2 = np.arange(min(data1), max(data1), 0.00001), np.arange(min(data2), max(data2), 0.00001)
    y1, y2 = 1/np.sqrt(2*np.pi)/sigma1 * np.exp(-(x1-mu1)**2/2/sigma1**2), 1/np.sqrt(2*np.pi)/sigma2 * np.exp(-(x2-mu2)**2/2/sigma2**2)
    # print(mu1, sigma1)

    ax1, ax2 = plt.subplot(121), plt.subplot(122)
    ax1.hist(data1, bins=30, density=False)
    # ax1.plot(x1, y1, color='r')
    ax2.hist(data2, bins=30, density=False)
    # ax2.plot(x2, y2, color='r')

    plt.savefig(dir+'/hist.jpg')


def KDD():
    x = np.array([2,3,5])
    y1 = np.array([9.54, 7.62, 9.21])  #U0
    y1_tr = np.array([1.57, 1.71, 1.60])
    y2 = np.array([9.09, 9.21, 10.1])  #U
    y2_tr = np.array([1.63, 1.71, 1.69])
    y3 = np.array([0.064, 0.0627, 0.0625])  #alpha
    y3_tr = np.array([0.002976, 0.003212, 0.002927])

    plt.figure()
    plt.plot(x, y1, marker='v', color='b')
    plt.ylim(7.5, 10)
    plt.xlabel('cutoff distance')
    plt.ylabel('MAE')
    plt.set_title('U0')

    plt.savefig('U0.jpg')


def wer_param():
    # plt.figure(figsize=(20, 6))
    # plt.scatter(x=20.992, y=26.0, c='blue', marker='o', label='BiLSTM')
    # plt.scatter(x=14.688, y=26.2, c='blue', marker='v', label='TCN')
    # plt.scatter(x=30.958, y=25.4, c='blue', marker='s', label='TCN+BiLSTM')
    # plt.scatter(x=16.164, y=26.9, c='blue', marker='x', label='Transformer')
    # plt.scatter(x=16.186, y=21.9, c='red', marker='*', label='LCTE(ours)')
    # plt.xlabel('Number of Parameters (M)')
    # plt.ylabel('Word Error Rate')
    # plt.legend()
    # plt.savefig('param_wer.jpg')
    
    blstm_par = 20.992
    blstm_wer = 26.0

    tcn_par = 14.688
    tcn_wer = 26.2

    tcnblstm_par = 30.958
    tcnblstm_wer = 25.4 

    tf_par = 16.164
    tf_wer = 26.9

    lcte_par = 16.186
    lcte_wer = 21.9

    dpi = 600
    # fig, ax1 = plt.subplots(1, sharex=True)
    par = [blstm_par, tcn_par, tcnblstm_par, tf_par, lcte_par]
    wer = [blstm_wer, tcn_wer, tcnblstm_wer, tf_wer, lcte_wer]
    # category = [str(i) for i in range(len(mem))]
    method_name = ["BiSLTM", 'TCN', 'TCN+BiLSTM', 'Transformer', 'LCTE (ours)']
    marker = ['o', 'X', '^', 'P', '*']
    # sns.set_style("white")
    plt.rcParams['pdf.fonttype'] = 42
    sns.color_palette("Paired")
    df = pd.DataFrame({'par':par, 'wer':wer, "category":method_name})
    markers = [marker[i] for i in range(len(df["category"].unique()))]
    # sns.scatterplot(data = df, x='mem', y='acc', markers=['o', 's'])

    plt.grid(True, linestyle='--')
    ax1 = sns.lmplot(data=df, hue='category', x='par', y='wer', markers=markers, fit_reg=True, aspect=2, scatter_kws={"s": 275}, facet_kws={'legend_out': False, "despine": False})
    # for ax in ax1.axes.flatten():
        # ax.tick_params(axis='y', which='both', direction='out', length=4, left=True)
        # ax.grid(b=True, which='both', color='gray', linewidth=0.5)

    ax1._legend.set_title(None)
    plt.setp(ax1._legend.get_texts(), fontsize=24) # for legend text
    # plt.setp(ax1._legend.get_title(), fontsize='32') # for legend title
    # ax1.set(xlim=[0, 13])
    # ax1.set(ylim=[0.25,0.425])
    ax1.set_xlabels("Number of Parameters (M)", fontsize=14)
    ax1.set_ylabels("Word Error Rate %", fontsize=14)
    # ax1.set_xlabels("X Label",fontsize=30)
    # ax1.set_ylabels("Y Label",fontsize=20)
    # ax1._xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x)))
    # plt.set(fontsize=14)
    # ax1 = df.plot.scatter(x='mem', y='acc')
    # plt.show()
    h,l = ax1.ax.get_legend_handles_labels()
    # for lh in h:
    #     lh.fontsize = 24
    ax1._legend.remove()
    ax1.ax.legend(h,l, ncol=2, fontsize=9.8, markerscale=0.7) # you can specify any location parameter you want here
    plt.savefig("param_wer.pdf", pad_inches=0, dpi=1200)
    # plt.savefig("memory.png", pad_inches=0, dpi=300)
    # ratio = [1 - (mem[7] / i) for i in mem[0:7]]
    # print(ratio)

# if __name__ == '__main__':
#     x = np.array([4.3, 5.3, 6.3, 7.3, 8.3])
#     dev = np.array([21.6, 22.0, 21.7, 22.1, 22.4])
#     tes = np.array([22.9, 22.3, 22.3, 22.1, 22.4])

#     head_share_dev = 21.7 * np.ones(x.shape[0])
#     head_share_test = 21.91 * np.ones(x.shape[0])

#     head_spec_dev = 21.4 * np.ones(x.shape[0])
#     head_spec_test = 21.89 * np.ones(x.shape[0])

#     head_share_nostat_dev = 22.5 * np.ones(x.shape[0])
#     head_share_nostat_test = 22.8 * np.ones(x.shape[0])

#     head_spec_nostat_dev = 22.3 * np.ones(x.shape[0])
#     head_spec_nostat_test = 22.7 * np.ones(x.shape[0])

#     fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(6.4,3.5))
#     ax1.plot(x, dev, color='blue', marker='v', label='fixed')
#     ax1.plot(x, head_share_dev, '--', color='green', label='h-sh')
#     ax1.plot(x, head_share_nostat_dev, ':', color='green', label='h-sh (w/o stat)')
#     ax1.plot(x, head_spec_dev, '--', color='red', label='h-sp')
#     ax1.plot(x, head_spec_nostat_dev, ':', color='red', label='h-sp (w/o stat)')
#     ax1.set_xlabel('window size')
#     ax1.set_ylabel('WER% (Dev)')
#     ax1.set_ylim(21.35, 22.6)
#     # ax1.legend(loc='upper left')

#     ax2.plot(x, tes, color='blue', marker='v', label='fixed')
#     ax2.plot(x, head_share_test, '--', color='green', label='h-sh')
#     ax2.plot(x, head_share_nostat_test, ':', color='green', label='h-sh (w/o stat)')
#     ax2.plot(x, head_spec_test, '--', color='red', label='h-sp')
#     ax2.plot(x, head_spec_nostat_test, ':', color='red', label='h-sp (w/o stat)')
#     ax2.set_xlabel('window size')
#     ax2.set_ylabel('WER% (Test)')
#     ax2.set_ylim(21.8, 23.0)
#     # ax2.legend(loc='upper right', bbox_to_anchor=(0,0))

#     handles, labels = ax2.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.7,0.75), fontsize=9, markerscale=0.7, labelspacing=0.1)

#     #plt.ylim(22.8, 24.0)
#     plt.tight_layout()
#     plt.savefig('fine_tune.pdf')
    
    # wer_ratio_fig('/2tssd/rzuo/codes/setr/results/cnntr2_gau_lea_D10.0_conv_cas_nobias')


# if __name__ == '__main__':
    # dir_lst = ['/2tssd/rzuo/codes/lcasan/results/lcasan_rpe_gau_modDfromQ_div2_qk10_cas']
              #'/2tssd/rzuo/codes/setr/results/cnntr2_gau_lea_D4.0_conv_gate']
    # gate_wer_fig(dir_lst)
    # dir_spec = '/2tssd/rzuo/codes/lcasan/results/lcasan_vgg11_rpe_gau_head_spec_qk10_2std'
    # dir_share = '/2tssd/rzuo/codes/lcasan/results/lcasan_vgg11_rpe_gau_D6.3_head_share_qk10_div2_2std'
    # dir_spec = '/2tssd/rzuo/codes/lcasan/results/lcasan_vgg11_rpe_gau_head_spec_2std'
    # dir_share = '/2tssd/rzuo/codes/lcasan/results/lcasan_vgg11_rpe_gau_head_share'
    # D_wer_fig(dir_spec, dir_share)
    # hist('/2tssd/rzuo/codes/lcasan/phoenix_datasets/stat_csl1_seganddrop.npz')

    # sc_box(['/2tssd/rzuo/codes/lcsa/results/vgg11_cbs_cbam4_noch_maxsoft_superatt_sc_tfmer_ape_dtcn_batch', '/2tssd/rzuo/codes/lcsa/results/vgg11_cbs_sc_tfmer_ape_dtcn_frame'])
    # hist_cw('/2tssd/rzuo/codes/lcsa/results/vgg11_cas_bef_san_cbam4_nochannel_maxsoftmax__superatt')
    # wer_param()


if __name__ == '__main__':
    dev = np.array([34.3, 35.1, 35.3, 33.1, 33.5, 35.0, 34.4])
    test = np.array([34.4, 33.8, 33.1, 32.7, 32.8, 34.2, 33.6])
    x = np.arange(0, 1.6, 0.25)

    fig, ax = plt.subplots(1,1)
    ax.plot(x, dev, color='blue', marker='v', label='dev')
    ax.plot(x, test, color='red', marker='v', label='test')
    ax.set_xlim(-0.1, 1.6)
    ax.set_ylim(32.5, 35.5)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('WER%')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.savefig('fine_tune_SI5.pdf')