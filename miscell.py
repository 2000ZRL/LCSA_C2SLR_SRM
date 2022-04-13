# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:36:04 2020

@author: 14048
"""
import torch as t
import torch.nn as nn
import numpy as np
import os
import cv2
import yaml
import math, random
# from multiprocessing import Pool
# from model import SLRModel
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
seed=10
gpu=1
t.manual_seed(seed)
t.cuda.manual_seed(seed)
t.cuda.manual_seed_all(seed)

class net1(nn.Module):
    def __init__(self, grad=True):
        super(net1, self).__init__()
        self.layer1 = nn.Linear(5,5)
        self.layer1.weight.requires_grad=grad
        self.layer2 = nn.Linear(5,5)
        self.layer3 = nn.Linear(5,5)
        self.layer4 = nn.Linear(5,5)
        self.num_iter = 0
    
    def forward(self, x):
        self.num_iter += 1
        iden = x
        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        return x

class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.layer = nn.Linear(5,1)
    
    def forward(self, x):
        return self.layer(x)

# net = net1()
# epoch = 10
# x = t.rand(2,5)
# print(net.num_iter)
# for i in range(epoch):
#     y = net(x)
# print(net.num_iter)

def get_lambda(i):
    # get lambda for gradient reversal layer
    gamma = 10
    total_iter = 4376 / 2 * 45
    p = i / total_iter
    return 2 / (1+math.exp(-gamma*p)) - 1

# lst = [i for i in range(4376 // 2 * 45)]
# res = []
# for i in lst:
#     res.append(get_lambda(i))
# lst = np.array(lst)
# res = np.array(res)
# plt.plot(lst, res)
# plt.xlabel('iter')
# plt.ylabel('lambda')
# plt.savefig('lambda.jpg')

def lin_normalize(hmaps):
    min = hmaps.min(axis=(-2,-1), keepdims=True)
    max = hmaps.max(axis=(-2,-1), keepdims=True)
    hmaps = (hmaps - min) / (max - min + 1e-6)
    return hmaps

def gen_gaussian_hmap(coords, hmap_shape, sigma=2):
    H, W = hmap_shape
    x, y = t.meshgrid(t.arange(H), t.arange(W))
    grid = t.stack([x,y], dim=2)  #[H,H,2]
    hmap = t.exp(-((grid-coords*(H-1))**2).sum(dim=-1) / (2*sigma**2))  #[H,H]
    return hmap


# a = np.load('results/SI_vgg11_cbs_cbam4_noch_maxsoft_superatt_sc_tfmer_ape_dtcn_batch/sg.npz')
# b = np.load('results/COR_SI_C2SLR_presigner_xvec_gradrev1.0/sg.npz')
# a = a['arr_0']
# b = b['arr_0']
# assert a.shape == b.shape
# print(np.abs(a-b).mean())

# generate default config
# default = {'data': {'dataset': '2014',
#                     'aug_type': 'random_drop',
#                     'max_len': 150,
#                     'p_drop': 0.5,
#                     'resize_shape': [256,256],
#                     'crop_shape': [224,224]},
#             'model': {'name': 'lcsa',
#                       'batch_size': 2,
#                       'emb_size': 512,
#                       'vis_mod': 'vgg11',
#                       'seq_mod': 'transformer'},
#             'dcn': {'ver': 'v2',
#                     'lr_factor': 1.0},
#             'transformer':{'tf_model_size': 512,
#                            'tf_ff_size': 2048,
#                            'num_layers': 2,
#                            'num_heads': 8,
#                            'dropout': 0.1,
#                            'emb_dropout': 0.1,
#                            'pe': 'rpe_gau',
#                            'D_std_gamma': [6.3, 1.4, 2.0],
#                            'mod_D': None,
#                            'mod_src': 'Q'},
#             'comb_conv': None,
#             'qkv_context': [1, 0, 0],

#             'va': 0,
#             've': 0,
#             'sema_cons': None,
#             'att_idx_lst': [],
#             'spatial_att': None,
#             'pool_type': 'avg',
#             'cbam_pool': 'max_avg',
#             'att_sup_type': 'first',
            
#             'pose': None,
#             'pose_arg': [3, 0.5],
#             'pose_dim': 0,
#             'heatmap_num': 3,
#             'heatmap_shape': [28],
#             'heatmap_type': 'gaussian',
#             'pose_f': 1.0,

#             'from_ckpt': 0,
#             'save_dir': './results',
#             'max_num_epoch': 60,
#             'num_pretrain_epoch': 0,

#             'optimizer':{'name': 'adam',
#                          'lr': 1e-4,
#                          'betas': (0.9, 0.999),
#                          'weight_decay': 1e-4,
#                          'momentum': None},

#             'lr_scheduler': {'name': 'plateau',
#                              'decrease_factor': 0.7,
#                              'patience': 6},
            
#             'beam_size': 10,
#             'seed': 42,
#             'gpu': 1,
#             'setting': 'full',
#             'mode': 'train',
#             'test_split': 'test'
#             }

# with open('default.yml', 'w') as f:
#     yaml.dump(default, f, default_flow_style=False, sort_keys=False)


def k_means(x, center, max_steps=10):
    # x: n*d, center: k*d
    n, d = x.shape
    k = center.shape[0]
    tx = np.tile(x, (k,1))  #[n*k,d]
    for s in range(max_steps):
        print('step: ', s)
        tcenter = np.tile(center, n).reshape(n*k, d)  #[n*k,d]
        dist = ((tx-tcenter)**2).sum(axis=-1)  #[n*k]
        dist = dist.reshape(k,n).T  #[n,k]
        cluster_idx = dist.argmin(axis=-1)  #[n]
        
        # update centers
        for i in range(k):
            data = x[cluster_idx==i]
            if data.size > 0:
                center[i] = data.mean(axis=0)
        print(dist, cluster_idx, center)
    
    return center, cluster_idx


# x = np.array([[55,50], [43,50], [55,52], [43,54], [58,53], [41,47], [50,41], [50,70]], dtype=np.float32)
# center = np.array([[43,50], [55,50]], dtype=np.float32)
# cluster, center = k_means(x, center, 2)

# with open('../../data/csl-daily/csl2020ct_v1.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data.keys())
# print(data['info'][0]['name'])
# print(len(data['gloss_map']))
# info = data['info']
# df = pd.DataFrame(info)
# # df = df.sort_values(by=['name'])
# print(df[0:1])

# spl = pd.read_csv('../../data/csl-daily/split_1.txt', sep="|")
# # spl = spl.sort_values(by=['name'])
# print(spl)

# df = df.merge(spl, how='inner', on='name')
# print(df)


# vis attention
# def vis_attention(self, model_file):
#     import cv2
#     from torchvision.transforms.functional import resize
#     state_dict = update_dict(t.load(model_file)['mainstream'])
#     self.model.load_state_dict(state_dict, strict=True)
#     self.model.cuda()
#     self.model.eval()
    
#     dset = self.create_dataloader(split='test', bsize=1)
#     idx = 5  # ATTENTION: random drop issue
#     for i, batch_data in tqdm(enumerate(dset), desc='[VIS_ATT phase]'):
#         video = t.cat(batch_data['video']).cuda()
#         len_of_video = batch_data['len_video'].cuda()
#         coord = t.cat(batch_data['coord'])
#         coord = coord[idx, ...]  #[3,2]
#         heatmap = []
#         for hmap in zip(*batch_data['heatmap']):
#             heatmap.append(t.cat(hmap).cuda()) 
        
#         op_dict = self.model(video, len_of_video, None)
#         offset_lst, mask_lst = op_dict['dcn_offset'], op_dict['spat_att']
        
#         if self.args.spatial_att in ['dcn', 'dcn_nooff']:
#             offset_lst = offset_lst[:2]  #first two layer offsets
#             H = offset_lst[1].shape[-1]
#             corners = [t.LongTensor([H//8,H//8]), t.LongTensor([H-H//8,H//8]), t.LongTensor([H//8,H-H//8]), t.LongTensor([H-H//8,H-H//8])]
#             mask_labels = gen_mask_from_pose(offset_lst[0], heatmap[0]).detach()
            
#             # visualize offsets
#             for j in range(2):
#                 offset_lst[j] = offset_lst[j][idx, ...]  #[18,28,28]
#             for j in range(9):
#                 offset = offset_lst[0]
#                 if j<3:
#                     center = ((H-1) * coord[j, ...]).long()
#                 elif j==3:
#                     center = t.LongTensor([H//2,H//2])
#                 elif j==4:
#                     center = t.LongTensor([H-3,H//2])
#                 else:
#                     center = corners[j-5]
#                 x, y = offset.split(9, dim=0)  #[9,28,28]
#                 img = video[idx, ...]  #[3,224,224]
#                 img = img.permute(1,2,0).cpu() + t.Tensor([0.5372,0.5273,0.5195])
#                 img = resize(img.permute(2,0,1), [H,H])
#                 img = img.permute(1,2,0).numpy().copy()
#                 img = (255*img).astype(np.uint8)[..., ::-1]
#                 img = cv2.UMat(img).get()
            
#                 first = []
#                 off_lst = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
#                 for k in range(9):
#                     tmp = center + t.Tensor(off_lst[k]) + t.Tensor((x[k,center[0],center[1]], y[k,center[0],center[1]]))
#                     first.append(tmp.long())
#                     if k!=4:
#                         cv2.circle(img, (int(tmp[1]), int(tmp[0])), radius=2, color=(0,0,255))
#                     else:
#                         cv2.circle(img, (int(tmp[1]), int(tmp[0])), radius=2, color=(255,0,0))
                
#                 # offset = offset_lst[0]
#                 # x, y = offset.split(9, dim=0)
#                 # for cen in first:
#                 #     for k in range(9):
#                 #         tmp = cen + t.Tensor(off_lst[k]) + t.Tensor((x[k,cen[0],cen[1]], y[k,cen[0],cen[1]]))
#                 #         tmp = tmp.clamp(0, H-1)
#                 #         cv2.circle(img, (int(tmp[1]), int(tmp[0])), radius=1, color=(0,0,255))
                
#                 print(center)
#                 cv2.circle(img, (center[1],center[0]), radius=1, color=(0,255,0))
#                 path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_vis_deform_{:d}_{:d}.jpg'.format(H,j))
#                 cv2.imwrite(path, img)
                
#             # visualize center kernel position
#             center_offset_x, center_offset_y = offset[4, ...], offset[13, ...]
#             c = t.stack(t.meshgrid(t.arange(H), t.arange(H)), dim=0).view(2,-1)
#             img = video[idx, ...]  #[3,224,224]
#             img = img.permute(1,2,0).cpu() + t.Tensor([0.5372,0.5273,0.5195])
#             img = resize(img.permute(2,0,1), [H,H])
#             img = img.permute(1,2,0).numpy().copy()
#             img = (255*img).astype(np.uint8)[..., ::-1]
#             img = cv2.UMat(img).get()
#             for j in range(H**2):
#                 x, y = c[:, j]
#                 o_x, o_y = center_offset_x[x, y], center_offset_y[x, y]
#                 cv2.circle(img, (int(x+o_x), int(y+o_y)), radius=1, color=(255,0,0))
#             path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_vis_deform_center_offsets.jpg')
#             cv2.imwrite(path, img)
        
#         # visualize mask
#         if self.args_model['vis_mod'] == 'mb_v2':
#             mask_lst = mask_lst[1:]
#         elif self.args_model['vis_mod'] == 'resnet18':
#             mask_lst = mask_lst[2:]
#         else:
#             mask_lst = mask_lst[0:]
#         m_i = 0
#         for masks in mask_lst:
#             if masks is None:
#                 continue
#             m_i += 1
#             if self.args.spatial_att in ['dcn', 'dcn_nooff']:
#                 for j in range(9):
#                     mask = masks[idx, j, ...]  #[28,28]
#                     mask_label = mask_labels[idx, j, ...]
#                     print(mask.max(), mask_label.max(), mask.min(), mask_label.min())
                    
#                     mask = (255*mask).detach().cpu().numpy().astype(np.uint8)
#                     # mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
#                     path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_mask_{:d}_{:d}.jpg'.format(m_i, j))
#                     cv2.imwrite(path, mask)
                    
#                     mask_label = (255*mask_label).cpu().numpy().astype(np.uint8)
#                     # mask_label = cv2.applyColorMap(mask_label, cv2.COLORMAP_JET)
#                     path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_mask_label_{:d}.jpg'.format(j))
#                     cv2.imwrite(path, mask_label)
#             else:
#                 mask = masks[idx, ...]
#                 if self.args.spatial_att == 'ca':
#                     mask = mask.mean(dim=0)
#                 mask = mask.squeeze()
#                 mask_label = heatmap[0][idx, 0, ...]
#                 print(mask.max(), mask_label.max(), mask.min(), mask_label.min())
                
#                 mask = (255*mask).detach().cpu().numpy().astype(np.uint8)
#                 path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_mask_{:d}.jpg'.format(m_i))
#                 cv2.imwrite(path, mask)
                
#                 mask_label = (255*mask_label).cpu().numpy().astype(np.uint8)
#                 path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_mask_label.jpg')
#                 cv2.imwrite(path, mask_label)
        
#         if self.args.cbam_pool in ['softmax', 'max_softmax']:
#             # distribution of channel weight
#             count = 0
#             for c_w in offset_lst:
#                 if c_w is None:
#                     continue
#                 count += 1
#                 c_w = c_w.squeeze().detach().cpu().numpy()[idx, :]  #[512]
#                 x = np.arange(0, c_w.shape[0])
#                 plt.figure(count)
#                 plt.plot(x, c_w, label='weight of channels')
#                 path = os.path.join(self.args.save_dir, self.args.save_dir.split('/')[-1]+'_weights_{:d}.jpg'.format(count))
#                 plt.savefig(path)



# def HITS(M, num_step):
#     N = M.shape[0]
#     h = np.ones([N,1])
#     a = np.ones([N,1])
#     for _ in range(num_step):
#         h = np.matmul(M, np.matmul(M.T, h))
#         a = np.matmul(M.T, np.matmul(M, a))

#         # normalize
#         h = N * h / h.sum()
#         a = N * a / a.sum()
#         print('h: ', h, 'a: ', a)

# M = np.array([[0,1,1,0], [0,0,0,0], [0,0,0,1], [0,0,1,0]], dtype=np.float32)
# HITS(M, 50)

name = 'S003181_P0001_T00'
dir = '../../data/csl-daily/'+name
image=cv2.imread(dir+'/000000.jpg')
height=image.shape[0]
width=image.shape[1]
size=(height,width)
fps=30
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter('badcase/'+name+'.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)) #创建视频流对象-格式一

files = os.listdir(dir)
files.sort()
for f in files:
    fname = dir+'/'+f
    image=cv2.imread(fname)
    video.write(image)  # 向视频文件写入一帧--只有图像，没有声音

# with open('../../data/csl-daily/csl2020ct_v1.pkl', 'rb') as f:
#     data = pickle.load(f)
# i = 0
# for item in data['info']:
#     print(item)
#     i += 1
#     if i==500:
#         break

# data = np.load('/2tssd/rzuo/codes/lcsa/results/CVPR22_C2SLR/vgg11_cbs_sc_tfmer_ape_dtcn_batch/sc_id_dist.npz')
# ids = list(data.keys())
# dist = list(data.values())
# diff = []
# for id in ids:
#     diff.append(data[id][1] - data[id][0])
# diff = np.array(diff)
# index = diff.argsort()
# # print(ids[index[-10:]])
# i = index[-1]
# if i%2 == 0:
#     print(ids[i], diff[i], dist[i], ids[i+1], diff[i+1], dist[i+1])
# else:
#     print(ids[i], diff[i], dist[i], ids[i-1], diff[i-1], dist[i-1])

# base_dir = "/2tssd/rzuo/data/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual"
# split = ['train', 'dev', 'test']
# for s in split:
#     fname = base_dir + '/' + s + '.SI5.corpus.csv'
#     df = pd.read_csv(fname, sep="|")
#     df["folder"] = df["folder"].apply(lambda x: s+'/'+x)
#     df.to_csv(base_dir+'/new_'+s+'.SI5.corpus.csv', sep="|", index=False)

# df_train = pd.read_csv(base_dir+'/new_train.SI5.corpus.csv', sep='|')
# df_dev = pd.read_csv(base_dir+'/new_dev.SI5.corpus.csv', sep='|')
# df_test = pd.read_csv(base_dir+'/new_test.SI5.corpus.csv', sep='|')
# df = pd.concat([df_train, df_dev, df_test])

# df_SI3 = df[df['signer'] == 'Signer03']
# df_train = df[df['signer'] != 'Signer03']
# df_SI3 = df_SI3.sample(frac=1)
# df_dev = df_SI3[:df_SI3.shape[0]//2]
# df_test = df_SI3[df_SI3.shape[0]//2:]
# print(df_train.shape, df_dev.shape, df_test.shape)

# df_train.to_csv(base_dir+'/new_train.SI3.corpus.csv', sep="|", index=False)
# df_dev.to_csv(base_dir+'/new_dev.SI3.corpus.csv', sep="|", index=False)
# df_test.to_csv(base_dir+'/new_test.SI3.corpus.csv', sep="|", index=False)

# df_dev = pd.read_csv(base_dir+'/new_dev.SI3.corpus.csv', sep='|')
# df_test = pd.read_csv(base_dir+'/new_test.SI3.corpus.csv', sep='|')
# with open('phoenix2014-SI3-groundtruth-dev.stm', 'w') as f:
#     for index, row in df_dev.iterrows():
#         f.write(row['id']+' 1 signer03 0.0 1.79769e+308 '+row['annotation']+'\n')

# with open('phoenix2014-SI3-groundtruth-test.stm', 'w') as f:
#     for index, row in df_test.iterrows():
#         f.write(row['id']+' 1 signer03 0.0 1.79769e+308 '+row['annotation']+'\n')


# base_dir = '../../data/csl-daily'
# path = base_dir + '/split_1.txt'
# spl = pd.read_csv(path, sep="|")
# with open(base_dir+'/csl2020ct_v1.pkl', 'rb') as f:
#     data = pickle.load(f)
# df = pd.DataFrame(data['info'])
# df = df.merge(spl, how='inner', on='name')
# df = df.rename(columns={"name": "id", "label_gloss": "annotation"})

# with open(path, 'r') as f:
#     lines = f.readlines()

# for i in [8,3,9,4,7,2,6]:
#     # first find the samples with that signer, then shuffle, split into dev and test
#     d_and_t = []
#     for l in lines:
#         if 'P000'+str(i) in l:
#             d_and_t.append(l)
#     num = len(d_and_t)
#     print(i, num)
#     random.shuffle(d_and_t)
#     dev = d_and_t[:num//2]
#     test = d_and_t[num//2:]

#     # second build new split
#     fname = 'split_SI{:s}.txt'.format(str(i))
#     with open(fname, 'w') as f:
#         for l in lines:
#             name, split = l.split('|')
#             if name == 'name':
#                 #first line
#                 f.write(l)
#                 continue
#             if 'P000'+str(i) in name:
#                 if l in dev:
#                     f.write(name+'|'+'dev\n')
#                 else:
#                     f.write(name+'|'+'test\n')
#             else:
#                 f.write(name+'|'+'train\n')

# for i in range(10):
#     # print(i, df[df['signer']==i].shape[0])
#     print(i, df[df['signer']==i].reindex().iloc[0]['id'])
# print('total', df.shape[0])

# check
# for i in [8,3,9,4,7,2,6]:
#     fname = 'split_SI{:s}.txt'.format(str(i))
#     with open(fname, 'r') as f:
#         lines = f.readlines()
#     train = dev = test = 0
#     for l in lines:
#         if 'dev' in l:
#             dev += 1
#         elif 'test' in l:
#             test += 1
#         else:
#             train += 1
#     print(i, train, dev, test, dev+test, train+dev+test)
# origin = base_dir + '/split_1.txt'
# with open(origin, 'r') as f:
#     lines = f.readlines()
# for i in [8,3,9,4,7,2,6]:
#     fname = base_dir + '/split_SI{:d}.txt'.format(i)
#     with open(fname, 'r') as f:
#         si = f.readlines()
#         for a,b in zip(lines, si):
#             a = a.split('|')[0]
#             b = b.split('|')[0]
#             if(a!=b):
#                 print(i, b)


# badcase analysis
# base = '/2tssd/rzuo/codes/lcsa/results/CVPR22_C2SLR/csl_daily_vgg11_cbs_sac_sec'
# fname = base + '/badcase.txt'
# with open(fname, 'r') as f:
#     lines = f.readlines()

# SUB, DEL, INS = [], [], []
# num_sub = num_del = num_ins = num_unk = 0
# for l in lines:
#     if 'SUB' in l:
#         num_sub += 1
#         SUB.append(l.split(' ')[1])
#         # SUB.append(set(l.strip().split(' ')[1:]))
#     elif 'DEL' in l:
#         num_del += 1
#         DEL.append(l.split(' ')[1])
#     elif 'INS' in l:
#         num_ins += 1
#         INS.append(l.split(' ')[1])
#     if 'unk' in l:
#         num_unk += 1
# num_err = num_sub + num_del + num_ins
# oov = num_unk / num_err

# sub_pair_count = {}
# for x in SUB:
#     sub_pair_count[str(x)] = sub_pair_count.get(str(x), 0)+1
# ordered = sorted(sub_pair_count.items(), key=lambda x:x[1], reverse=True)
# with open('csld_sub_pairs.txt', 'w') as f:
#     f.write('Pairs Times\n')
#     for i in ordered:
#         f.write(i[0] + ' ' + str(i[1]) + '\n')

# with open('phoenix_datasets/gls_count_CSLD_train.pkl', 'rb') as f:
#     gls_count_train = pickle.load(f)
# with open('phoenix_datasets/gls_count_CSLD_test.pkl', 'rb') as f:
#     gls_count_test = pickle.load(f)
# print('error by oov: ', oov, 'test glosses: ', sum(gls_count_test.values()))
# tot = sum(gls_count_train.values())
# for g in gls_count_train.keys():
#     gls_count_train[g] /= tot
# print(max(gls_count_train.values()), min(gls_count_train.values()), np.array(list(gls_count_train.values())).mean())

# SUB.extend(DEL)
# gls = set(SUB)
# train_freq = []
# test_err_rate = []
# freq = {}
# for g in gls:
#     num = 0
#     for s in SUB:
#         if s == g:
#             num += 1
#     test_err_rate.append(num / gls_count_test[g])
#     freq[g] = num / gls_count_test[g]
#     try:
#         train_freq.append(gls_count_train[g])
#     except:
#         train_freq.append(0.0)
# train_freq, test_err_rate = np.array(train_freq), np.array(test_err_rate)
# idx = np.argsort(train_freq)[::-1]
# test_err_rate = test_err_rate[idx]
# x = np.arange(test_err_rate.shape[0])
# plt.bar(x, test_err_rate, width=5)
# plt.xlabel('gloss sorted by training frequency')
# plt.ylabel('test error rate')
# plt.savefig('2014_freq.jpg')

# f = np.sort(train_freq)[::-1]
# x = np.arange(f.shape[0])
# plt.bar(x, f, width=5)
# plt.xlabel('gloss sorted by training frequency')
# plt.ylabel('training frequency')
# plt.savefig('2014_train_freq.jpg')


# digit_tr = []
# digit_test = []
# for g in gls:
#     if g.isdigit():
#         digit_tr.append(gls_count_train[g])
#         digit_test.append(freq[g])
#         print(g, gls_count_train[g], freq[g])
# digit_tr = np.array(digit_tr)
# digit_test = np.array(digit_test)
# plt.scatter(digit_tr, digit_test)
# plt.xlabel('training frequency')
# plt.ylabel('test error rate')
# plt.savefig('digit.jpg')