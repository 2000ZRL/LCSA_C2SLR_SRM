# import sys
# sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch; torch.backends.cudnn.enabled = False
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from raft.raft_model import RAFT
from raft.utils import InputPadder

from phoenix_datasets import PhoenixTVideoTextDataset, CSLDailyVideoTextDataset
from tqdm import tqdm


def create_dataloader(dset_name='2014', split='train', bsize=1):
    dset_dict = {'2014T': {'cls': PhoenixTVideoTextDataset, 'root': '../../data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T', 'mean': [0,0,0], 'hmap_mean': []},
                 'csl-daily': {'cls': CSLDailyVideoTextDataset, 'root': '../../data/csl-daily', 'mean': [0,0,0]}}    
    args_data = {'dataset': dset_name, 'aug_type': 'random_drop', 'max_len': 999, 'p_drop': 0, 'resize_shape': [256,256], 'crop_shape': [256,256]}

    dset_dict = dset_dict[dset_name]
    dset = dset_dict['cls'](args=args_data,
                            root=dset_dict['root'],
                            split=split,
                            normalized_mean=dset_dict['mean'],
                            use_random=False,
                            temp_scale_ratio=0)
    
    dataloader = DataLoader(dset, 
                            bsize,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=dset.collate_fn,
                            drop_last=False)
    
    return dset_dict['root'], dataloader


@torch.no_grad()
def validate(model, args, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()

    _, valid_loader = create_dataloader(dset_name=args.dataset, split=args.split, bsize=1)

    if args.dataset == '2014T':
        path = os.path.join('/3tdisk/shared/rzuo/PHOENIX-2014-T', 'flow_things', args.split)
    elif args.dataset == 'csl-daily':
        path = os.path.join('/3tdisk/shared/rzuo/CSL-Daily', 'flow_things')
    if not os.path.exists(path):
        os.makedirs(path)

    for i, batch_data in tqdm(enumerate(valid_loader), desc='[Generating optical flow of {:s} of {:s}]'.format(args.split, args.dataset)):
        video = torch.cat(batch_data['video']).cuda()  #[T,3,256,256]
        # video = torch.rand(250,3,256,256).cuda()
        len_video = batch_data['len_video'][0]
        video_id = batch_data['id'][0]
        fname = os.path.join(path, ''.join(video_id))

        # if ''.join(video_id)+'.npz' in os.listdir(path):
        #     continue

        clip_lst = video.split(args.max_num_frame, dim=0)  # avoid OOM
        flow_lst = []
        i = 0
        num_clip = len(clip_lst)
        for v in clip_lst:
            if v.shape[0] > 1:
                # inputs of raft is uint8 then float
                v1 = (255*v[:-1, ...]).int().float()
                v2 = (255*v[1:, ...]).int().float()

                padder = InputPadder(v1.shape)
                v1, v2 = padder.pad(v1, v2)

                flow_low, flow_pr = model(v1, v2, iters=iters, test_mode=True)
                # flow = padder.unpad(flow_pr).cpu().detach().numpy()
                flow_low = flow_low.cpu().detach().numpy()
                flow_lst.append(flow_low)
            
            # deal with the last frame
            if i < num_clip-1:
                v1 = (255*v[-1, ...]).int().float()[None]
                v2 = (255*clip_lst[i+1][0, ...]).int().float()[None]
                v1, v2 = padder.pad(v1, v2)

                flow_low, flow_pr = model(v1, v2, iters=iters, test_mode=True)
                # flow = padder.unpad(flow_pr).cpu().detach().numpy()
                flow_low = flow_low.cpu().detach().numpy()
                flow_lst.append(flow_low)
            i += 1

        flow = np.concatenate(flow_lst, axis=0)
        assert flow.shape == (len_video-1, 2, 256//8, 256//8)
        np.savez_compressed(fname+'.npz', flow=flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default="/2tssd/rzuo/pretrained_models/RAFT_models/raft-things.pth")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--dataset', type=str, default='2014T', choices=['2014T', 'csl-daily'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--max_num_frame', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()

    with torch.no_grad():
        validate(model.module, args)

    # path_lst = ['PHOENIX-2014-T/flow_things/train', 'PHOENIX-2014-T/flow_things/dev', 'PHOENIX-2014-T/flow_things/test', 'CSL-Daily/flow_things']
    # base = '/3tdisk/shared/rzuo'
    # for p in path_lst:
    #     for f in tqdm(os.listdir(os.path.join(base, p))):
    #         data = np.load(os.path.join(base, p, f))['flow']
    #         data = data.astype(np.float16)
    #         np.savez_compressed(os.path.join(base, p, f), flow=data)