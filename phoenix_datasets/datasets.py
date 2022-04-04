import glob
import numpy as np
from sympy import S
import torch
from torch.utils.data import Dataset
from functools import partial
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from math import sqrt
import os


def load_pil(use_random, resize_shape, crop_shape, normalized_mean, path):
    # convert back to numpy as tensor in dataloader may cause
    # fd problem: https://github.com/pytorch/pytorch/issues/11201
    # and you probably don't want:
    # torch.multiprocessing.set_sharing_strategy('file_system')
    transform = transforms.Compose(
            [
                transforms.Resize(resize_shape),
                transforms.RandomCrop(crop_shape) if use_random else transforms.CenterCrop(crop_shape), 
                transforms.ToTensor(),
                transforms.Normalize(normalized_mean, [1, 1, 1])
            ]
        )
    return transform(Image.open(path)).numpy()


class VideoTextDataset(Dataset):
    Corpus = None

    def __init__(self, 
                 args,
                 root, 
                 normalized_mean, 
                 split, 
                 use_random,
                 temp_scale_ratio=0, 
                 vocab=None,
                 pose=None,
                 heatmap_shape=[56],
                 heatmap_num=7,
                 heatmap_mean=[],
                 heatmap_type='gaussian'):
        """
        Args:
            root: Root to the data set, e.g. the folder contains features/ annotations/ etc..
            split: data split, e.g. train/dev/test
            p_drop: proportion of frame dropping.
            random_drop: if True, random drop else evenly drop.
            vocab: gloss to index (categorize).
        """
        assert 0 <= args['p_drop'] <= 1, f"p_drop value {args['p_drop']} is out of range."
        assert (
            self.Corpus is not None
        ), f"Corpus is not defined in the derived class {self.__class__.__name__}."
        self.dset_name = args['dataset']
        self.aug_type = args['aug_type']
        self.max_len = args['max_len']
        
        self.corpus = self.Corpus(root, self.max_len)
        self.split = split
        if split == 'train_no_drop':
            self.split = 'train'
        self.resize_shape = args['resize_shape']
        self.crop_shape = args['crop_shape']
        self.normalized_mean = normalized_mean
        self.use_random = use_random  #True for train.
        self.p_drop = args['p_drop']
        self.temp_scale_ratio = temp_scale_ratio

        self.data_frame = self.corpus.load_data_frame(self.split)
        if split == 'train_no_drop':
            self.split = split  #go back to 'train_no_drop'
        self.vocab = vocab or self.corpus.create_vocab()
        self.pose = pose
        self.h_lst, self.heatmap_num, self.heatmap_mean, self.heatmap_type = heatmap_shape, heatmap_num, heatmap_mean, heatmap_type

    def sample_indices(self, n):
        if self.aug_type == 'random_drop':
            p_kept = min(1-self.p_drop, self.max_len/(n+1))
            if self.use_random:
                indices = np.arange(n)
                np.random.shuffle(indices)
                indices = indices[: int((n+1) * p_kept)]  #to make the length equal to that of no drop
                indices = sorted(indices)
            else:
                indices = np.arange(0, n, 1 / p_kept)  
                indices = np.round(indices)
                indices = np.clip(indices, 0, n-1)
                indices = indices.astype(int)
        
        elif self.aug_type == 'temp_scale':
            p_kept = 1 - self.temp_scale_ratio
            if self.use_random:
                indices = np.arange(n)
                np.random.shuffle(indices)
                indices = indices[: int((n+1) * p_kept)]  
                add_indices = indices[: int(n * self.temp_scale_ratio)]
                indices = np.concatenate((indices, add_indices))
                indices = sorted(indices)
            else:
                indices = np.arange(0, n)
                indices = indices.astype(int)
        
        elif self.aug_type == 'seg_and_drop':
            seg_len = 2
            p_kept = (1.0 - 1.0/seg_len) * min(1-self.p_drop, self.max_len/(n//seg_len+1))
            if self.use_random:                
                num_seg = n//seg_len
                indices = []
                for i in range(num_seg):
                    spl_lst = np.array([x for x in range(i*seg_len, (i+1)*seg_len)])
                    np.random.shuffle(spl_lst)
                    indices.append(spl_lst[:seg_len//2])
                if n % seg_len != 0:
                    spl_lst = np.array([x for x in range((i+1)*seg_len, n)])
                    np.random.shuffle(spl_lst)
                    indices.append(spl_lst[:seg_len//2])
                indices = np.concatenate(indices)
                # indices = sorted(indices)
                
                # drop part
                len_indices = indices.shape[0]
                p_kept = min(1-self.p_drop, self.max_len/(len_indices+1))
                np.random.shuffle(indices)
                indices = indices[: int((len_indices+1) * p_kept)]  #to make the length equal to that of no drop
                indices = sorted(indices)
            else:
                indices = np.arange(0, n, 1 / p_kept)  
                indices = np.round(indices)
                indices = np.clip(indices, 0, n-1)
                indices = indices.astype(int)
        
        else:
            raise ValueError('Wrong data augmentation type!')
        return indices

    @staticmethod
    def select_elements(l, indices):
        return [l[i] for i in indices]
    
    def gen_gaussian_hmap(self, coords, H):
        hmap_num = self.heatmap_num
        gamma = [14,14,14]
        sigma = [H/g for g in gamma]
        T = coords.shape[0]
        x, y = torch.meshgrid(torch.arange(H), torch.arange(H))
        grid = torch.stack([x,y], dim=2)  #[H,H,2]
        grid = grid.repeat((T,1,1,1)).permute(1,2,0,3)  #[H,H,T,2]
        hmap = [torch.exp(-((grid-(c.squeeze()*(H-1)))**2).sum(dim=-1) / (2*s**2)) for c,s in zip(coords.chunk(hmap_num, dim=1),sigma)]  #[H,H,T]
        hmap = torch.stack(hmap, dim=0).permute(3,0,1,2)  #[T,3,H,H]
        return hmap
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample = {**self.data_frame.iloc[index].to_dict()}  # copy
        frames = self.corpus.get_frames(sample, "fullFrame-210x260px")

        indices = self.sample_indices(len(frames))
        # print(sample['id'], len(frames))

        lp = partial(load_pil, self.use_random, self.resize_shape, self.crop_shape, self.normalized_mean)
        frames = self.select_elements(frames, indices)
        frames = np.stack(list(map(lp, frames)))
        
        # video level random crop
        # if self.use_random:
        #     crop = transforms.RandomCrop(self.crop_shape)
        # else:
        #     crop = transforms.CenterCrop(self.crop_shape)
        # frames = crop(torch.tensor(frames)).numpy()
        
        texts = list(map(self.vocab, sample["annotation"]))
        ids = list(sample['id'])
        
        hmaps = torch.empty(1)
        finer_coords = torch.empty(1)
        hmaps_norm = []
        if self.pose is not None:
            video_id = ''.join(ids)
            if self.dset_name in ['2014', '2014SI', '2014SI7']:
                fname = os.path.join('../../data/phoenix2014-release/phoenix-2014-multisigner/heatmaps_7', video_id+'.npz')
            elif self.dset_name == 'csl-daily':
                fname = os.path.join(self.corpus.root, 'heatmaps_7', video_id+'.npz')
            else:
                fname = os.path.join(self.corpus.root, 'heatmaps_7', self.split, video_id+'.npz')
            data = np.load(fname)
            if self.dset_name != 'csl-daily':
                heatmaps = data['heatmaps']
            finer_coords = data['finer_coords']
            # coords = torch.from_numpy(coords[indices, ...]).float()
            finer_coords = torch.from_numpy(finer_coords[indices, ...]).float()
            heatmap_mean = self.heatmap_mean[:]
            if self.heatmap_num == 3:
                idx = [0,6,1]
                if self.dset_name != 'csl-daily':
                    heatmaps = heatmaps[:, idx, ...]
                finer_coords = finer_coords[:, idx, ...]
                heatmap_mean = [self.heatmap_mean[i] for i in idx]
            
            if self.heatmap_type == 'origin':
                hmaps = [torch.from_numpy(heatmaps[indices, ...]).float()]
            elif self.heatmap_type in ['gaussian', 'norm']:
                hmaps = []
                for H in self.h_lst:
                    tmp = self.gen_gaussian_hmap(finer_coords, H)
                    if self.heatmap_type == 'norm':
                        # Directly L1 norm over spatial dimension, it is equal to softmax on negative distance
                        hmaps_norm.append(tmp / tmp.sum(dim=(-2,-1), keepdims=True))
                    hmaps.append(tmp)
            
            if self.pose == 'modality':
                for i in range(len(hmaps)):
                    hmap = transforms.functional.normalize(hmaps[i], heatmap_mean, [1]*self.heatmap_num)
                    hmap = transforms.functional.resize(hmap, [self.h_lst[i],self.h_lst[i]])
                    hmaps[i] = hmap.amax(dim=1, keepdim=True)
            
            elif self.pose in ['filter', 'deform_mask', 'deform_and_mask', 'super_att', 'prior']:
                for i in range(len(hmaps)):
                    hmap = transforms.functional.resize(hmaps[i], [self.h_lst[i],self.h_lst[i]])
                    hmaps[i] = hmap.amax(dim=1, keepdim=True)
                    if len(hmaps_norm) > 0:
                        hmaps_norm[i] = hmaps_norm[i].mean(dim=1, keepdims=True)
                
            elif self.pose in ['deform', 'deform_all', 'deform_patch', 'vit_patch']:
                # assert self.heatmap_num == 3
                hmaps = torch.empty(1)
            
            else:
                raise ValueError()
        
        sample.update(
            video=frames,
            text=texts,
            id=ids,
            heatmap=hmaps,
            heatmap_norm=hmaps_norm,
            coord=finer_coords)
        
        return sample
        # return {
        #     "video": frames,
        #     "text": texts,
        #     "id": ids
        # }

    @staticmethod
    def collate_fn(batch):
        collated = defaultdict(list)
        len_video = []
        len_label = []
        for sample in batch:
            collated["video"].append(torch.tensor(sample["video"]).float())
            collated["label"].extend(sample['text'])
            len_video.append(sample['video'].shape[0])
            len_label.append(len(sample['text']))
            collated['id'].append(sample['id'])
            collated['signer'].append(sample['signer'])
            collated['heatmap'].append(sample['heatmap'])
            collated['heatmap_norm'].append(sample['heatmap_norm'])
            collated['coord'].append(sample['coord'])
        collated['label'] = torch.LongTensor(collated['label'])
        collated['len_video'] = torch.LongTensor(len_video)
        collated['len_label'] = torch.LongTensor(len_label)
        return dict(collated)
    
    
    # def collate_fn(batch):
    #     #pad video and text
    #     len_video = [sample['video'].shape[0] for sample in batch]
    #     len_label = [len(sample['text']) for sample in batch]
    #     assert len(len_video) == len(len_label)
    #     batch_video = torch.zeros(len(len_video), 
    #                               max(len_video), 
    #                               batch[0]['video'][0].shape[-3],
    #                               batch[0]['video'][0].shape[-2],
    #                               batch[0]['video'][0].shape[-1])  #[B, max_len, 3, 260, 210]
    #     batch_label = []
    #     IDs = []
    #     for i in range(len(len_video)):
    #         batch_video[i, :len_video[i], ...] = torch.FloatTensor(batch[i]['video'])
    #         batch_label.extend(batch[i]['text'])
    #         IDs.append(batch[i]['id'])
            
    #     batch_label = torch.LongTensor(batch_label)
    #     len_video = torch.LongTensor(len_video)
    #     len_label = torch.LongTensor(len_label)
        
    #     return {'video': batch_video, 'label': batch_label, 'len_video': len_video,
    #             'len_label': len_label, 'id':IDs}
