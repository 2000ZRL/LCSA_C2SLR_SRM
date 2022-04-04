# -*- coding: utf-8 -*-
from phoenix_datasets import PhoenixVideoTextDataset, PhoenixTVideoTextDataset, PhoenixSIVideoTextDataset, PhoenixSI7VideoTextDataset,\
    CSLVideoTextDataset, CSLDailyVideoTextDataset, CFSWVideoTextDataset, TVBVideoTextDataset

from torch.utils.data import DataLoader
import torch as t
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os, pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tqdm import tqdm
from utils.metric import get_wer_delsubins

# root = "../../data/phoenix2014-release/phoenix-2014-multisigner"
# dtrain = PhoenixVideoTextDataset(
#     # your path to this folder, download it from official website first.
#     root=root,
#     split="train",
#     resize_shape=[256,256],
#     crop_shape=[224,224],
#     normalized_mean=[0,0,0],
#     aug_type='random_drop',
#     p_drop=0.5,
#     use_random=False
# )

# root = '/2tssd/rzuo/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
# dtrain = PhoenixTVideoTextDataset(
#     # your path to this folder, download it from official website first.
#     root=root,
#     split="train",
#     resize_shape=[256,256],
#     crop_shape=[224,224],
#     normalized_mean=[0,0,0],
#     aug_type='random_drop',
#     p_drop=0.5,
#     use_random=False
# )



args = {'dataset': '2014', 'aug_type': 'seg_and_drop', 'max_len': 140, 'p_drop': 0.4, 'resize_shape': [256,256], 'crop_shape': [256,256]}
# dtrain = CSLVideoTextDataset(args,
#                             root=('/2tssd/rzuo/data/ustc-csl', 'split_2.txt'),
#                             split='test',
#                             normalized_mean=[0,0,0],
#                             use_random=False
#                             )
# dtrain = CSLDailyVideoTextDataset(args,
#                                 root='/2tssd/rzuo/data/csl-daily',
#                                 split='test',
#                                 normalized_mean=[0,0,0],
#                                 use_random=False,
#                                 )
dtrain = PhoenixVideoTextDataset(args,
                                root='../../data/phoenix2014-release/phoenix-2014-multisigner',
                                split='test',
                                normalized_mean=[0,0,0],
                                use_random=False,
                                )
# dtrain = TVBVideoTextDataset(args, root='/6tdisk/shared/tvb', split='train', normalized_mean=[0,0,0], use_random=False)
# dtrain = PhoenixSI7VideoTextDataset(args, root='../../data/phoenix2014-release/phoenix-2014-signerindependent-SI5', split='train', normalized_mean=[0,0,0], use_random=False)

# root='/2tssd/rzuo/data/phoenix2014-release/phoenix-2014-signerindependent-SI5'
# dtrain = PhoenixSIVideoTextDataset(
#     # your path to this folder, download it from official website first.
#     root=root,
#     split="train",
#     resize_shape=[256,256],
#     crop_shape=[224,224],
#     normalized_mean=[0,0,0],
#     aug_type='random_drop',
#     p_drop=0.5,
#     use_random=True
# )

# dtrain = CFSWVideoTextDataset(root='/2tssd/rzuo/data/ChicagoFSWild',
#                               split='test',
#                               resize_shape=[256,256],
#                               crop_shape=[256,256],
#                               normalized_mean=[0,0,0],
#                               aug_type='random_drop',
#                               p_drop=0,
#                               use_random=False
#                               )

vocab = list(dtrain.vocab.table.keys())

print("Vocab", vocab)
print('vocab size', len(vocab))
print('len dtrain', len(dtrain))
# print('unk', list(dtrain.vocab.table.keys())[1232])
# print('blank', list(dtrain.vocab.table.keys())[1233])
# num_frame = np.zeros(len(dtrain))
# num_gls = np.zeros(len(dtrain))
# for gls in list(dtrain.vocab.table.keys()):
#     if 'unk'in gls or 'UNK' in gls:
#         print(gls)

bsize = 1
num_frame = np.zeros(len(dtrain)//bsize, dtype=np.float64)
num_gls = np.zeros(len(dtrain)//bsize, dtype=np.float64)
frame_mean = np.zeros((3, len(dtrain)//bsize), dtype=np.float64)
dl = DataLoader(dtrain, batch_size=bsize, collate_fn=dtrain.collate_fn, num_workers=20, shuffle=False)

f1 = 'results/CVPR22_C2SLR/vgg11_rpe_gau_D6.3_cas_bef_san/predictions.pkl'
f2 = 'results/CVPR22_C2SLR/vgg11_cas_bef_san_cbam4_nochannel_maxsoftmax__superatt/predictions.pkl'
f3 = 'results/CVPR22_C2SLR/vgg11_cbs_cbam4_noch_maxsoft_superatt_sc_tfmer_ape_dtcn_batch/predictions.pkl'
with open(f1, 'rb') as f:
    d1 = pickle.load(f)
with open(f2, 'rb') as f:
    d2 = pickle.load(f)
with open(f3, 'rb') as f:
    d3 = pickle.load(f)

ids = []
gloss = []
gls_count = {}
for i, batch in tqdm(enumerate(dl)):
    video = batch['video']  #a list of batch_size videos
    label = batch['label']  #a tensor
    len_video = batch['len_video']  #a tensor of length of each video
    len_label = batch['len_label']  #a tensor of length of each label
    video_id = ''.join(batch['id'][0])
    num_frame[i] += len_video[0]
    num_gls[i] += len_label[0]
    # video = t.cat(video)  #[sum_T,3,260,210]
    # gloss.extend(label.cpu().tolist())
    # print(video.shape)

    # video = video.permute(1,0,2,3).contiguous().view(3,-1)
    # frame_mean[:, i] = t.mean(video, dim=-1).numpy()
    # for name in video_id:
    #     ids.append(''.join(name))

    # for l in label:
    #     if vocab[l] not in gls_count.keys():
    #         gls_count[vocab[l]] = 0
    #     gls_count[vocab[l]] += 1
    
    # p1, p2, p3 = d1[video_id], d2[video_id], d3[video_id]
    # w1, w2, w3 = get_wer_delsubins(list(label), p1, True, vocab,1,1)[0], get_wer_delsubins(list(label), p2, True,vocab,1,1)[0], get_wer_delsubins(list(label), p3, True,vocab,1,1)[0]
    # if w1>w2 and w2>w3:
    #     print(video_id)
    #     with open('c2slr_compare.txt', 'a+') as f:
    #         f.write(video_id+'\n')
    #         f.write('REF: '+' '.join([vocab[x] for x in list(label)])+'\n')
    #         f.write('BAS: '+' '.join([vocab[x] for x in p1])+'\n')
    #         f.write('SAC: '+' '.join([vocab[x] for x in p2])+'\n')  
    #         f.write('C2SLR: '+' '.join([vocab[x] for x in p3])+'\n\n')  

# with open('phoenix_datasets/gls_count_2014_test.pkl', 'wb') as f:
#     pickle.dump(gls_count, f)
# gloss = np.unique(np.array(gloss))
# print(gloss.shape)
# real_mean = frame_mean * num_frame
# real_mean = np.sum(real_mean, axis=-1) / np.sum(num_frame)
# print('channel mean: ', real_mean)
# np.savez('./phoenix_datasets/stat_tvb_v4.1.npz', frame=num_frame, gls=num_gls, channel_mean=real_mean)

# stat = np.load('./phoenix_datasets/stat_2014T.npz')
# num_frame = stat['frame']
# num_gls = stat['gls']

ratio = num_frame / num_gls
# np.savez('./phoenix_datasets/stat_csldaily.npz', frame=num_frame, gls=num_gls)
ratio = np.sort(ratio)
num_frame = np.sort(num_frame)
num_gls = np.sort(num_gls)
print('total number of gls: ', np.sum(num_gls))
print('mean of frames per gls: ', np.mean(ratio), 'std of ratio: ', np.std(ratio, ddof=1))
print('mean of frames: ', np.mean(num_frame))
print('mean of gls: ', np.mean(num_gls))
print('top 5 ratio: ', ratio[-5:], ratio[:5])
print('top 5 frame length: ', num_frame[-5:], num_frame[:5])
print('top 5 gls length: ', num_gls[-5:], num_gls[:5])


# stat_frame = np.zeros(300)
# for num in num_frame:
#     stat_frame[int(num)] += 1

# stat_gls = np.zeros(30)
# for num in num_gls:
#     stat_gls[int(num)] += 1
    
# stat_ratio = np.zeros(55)
# i = 0
# for num in ratio:
#     stat_ratio[int(num)+1] += 1
    
# #draw
# x = np.arange(300)
# plt.figure(1)
# plt.plot(x, stat_frame, 'b')
# plt.xlabel('frame')
# plt.savefig('./phoenix_datasets/stat_frame.jpg')

# x = np.arange(30)
# plt.figure(2)
# plt.plot(x, stat_gls, 'b')
# plt.xlabel('gls')
# plt.savefig('./phoenix_datasets/stat_gls.jpg')

# x = np.arange(55)
# plt.figure(3)
# plt.plot(x, stat_ratio, 'b')
# plt.xlabel('ratio')
# plt.savefig('./phoenix_datasets/stat_ratio.jpg')

# x = np.arange(50)
# y = np.arange(50)
# plt.plot(x, y, 'bo-')
# plt.xlabel('x')
# plt.savefig('D:\\HKUST\\research\\codes\\setr\\x.jpg')

# print('avg_frame {:.3f}, max_frame {:.3f}, min_frame {:.3f}'\
#       .format(np.mean(num_frame), np.max(num_frame), np.min(num_frame)))
# print('avg_gls {:.3f}, max_gls {:.3f}, min_gls {:.3f}'\
#       .format(np.mean(num_gls), np.max(num_gls), np.min(num_gls)))
# np.savez(root+'/frame_gls_stat.npz', frame=num_frame, gls=num_gls)

# stat = np.load(root+'/frame_gls_stat.npz')
# total_frames = np.sum(stat['frame'])
# total_gls = np.sum(stat['gls'])
# avg = total_frames / total_gls
# print(avg)
