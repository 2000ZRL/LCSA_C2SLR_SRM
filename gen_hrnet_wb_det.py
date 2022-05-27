# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/kennymckormick/pyskl/blob/main/tools/data/custom_2d_skeleton.py
import argparse
import os
from unittest.mock import NonCallableMagicMock; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
from phoenix_datasets import PhoenixVideoTextDataset, PhoenixTVideoTextDataset, CSLDailyVideoTextDataset
from torch.utils.data import DataLoader, sampler
import torch; torch.hub.set_dir("/2tssd/rzuo/pretrained_models")
import cv2, time
from functools import partial

import decord
import mmcv
import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm

from pyskl.smp import mrlines

try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

default_mmdet_root = osp.dirname(mmdet.__path__[0])
default_mmpose_root = osp.dirname(mmpose.__path__[0])
default_det_config = (
    f'{default_mmdet_root}/mmdet/configs/faster_rcnn/'
    'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py')
default_det_ckpt = (
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
default_pose_config = (
    f'{default_mmpose_root}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/'
    'coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py')
default_pose_ckpt = (
    'https://download.openmmlab.com/mmpose/top_down/hrnet/'
    'hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth')


def create_dataloader(dset_name='2014T', split='train', bsize=1, sampler=None):
    dset_dict = {'2014T': {'cls': PhoenixTVideoTextDataset, 'root': '../../data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T', 'mean': [0,0,0], 'hmap_mean': []},
                '2014': {'cls': PhoenixVideoTextDataset, 'root': '../../data/phoenix2014-release/phoenix-2014-multisigner', 'mean': [0,0,0], 'hmap_mean': []},
                 'csl-daily': {'cls': CSLDailyVideoTextDataset, 'root': '../../data/csl-daily', 'mean': [0,0,0]}}
    if dset_name in ['2014', '2014T']:
        args_data = {'dataset': dset_name, 'aug_type': 'random_drop', 'max_len': 999, 'p_drop': 0, 'resize_shape': [260,210], 'crop_shape': [260,210]}
    elif dset_name == 'csl-daily':
        args_data = {'dataset': dset_name, 'aug_type': 'random_drop', 'max_len': 999, 'p_drop': 0, 'resize_shape': [512,512], 'crop_shape': [512,512]}

    dset_dict = dset_dict[dset_name]
    dset = dset_dict['cls'](args=args_data,
                            root=dset_dict['root'],
                            split=split,
                            normalized_mean=dset_dict['mean'],
                            use_random=False,
                            temp_scale_ratio=0)
    
    len_dset = len(dset)

    dataloader = DataLoader(dset, 
                            bsize,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=8,
                            collate_fn=dset.collate_fn,
                            drop_last=False)
    
    return len_dset, dataloader


def detection_inference(model, frames):
    results = []
    for frame in frames:
        result = inference_detector(model, frame[0])
        results.append(result)
    return results


def pose_inference(model, frames, det_results):
    if det_results is not None:
        assert len(frames) == len(det_results)
        total_frames = len(frames)
        kp = np.zeros((total_frames, 133, 3), dtype=np.float32)
        bb = np.zeros((total_frames, 5), dtype=np.float32)

        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
            pose = sorted(pose, key=lambda x:x['bbox'][-1])
            keypoints, bbox = pose[-1]['keypoints'], pose[-1]['bbox']
            kp[i] = keypoints
            bb[i] = bbox

    else:
        print(frames.shape)
        d = [{'bbox': np.array([0, 0, frames.shape[2]-1, frames.shape[1]-1])}]
        pose = inference_top_down_pose_model(model, frames[0], None, format='xyxy')[0]

    return kp


def detection_inference_one_frame(args, model, frame):
    raw_det_result = inference_detector(model, frame[0])
    # * Get detection results for human
    det_result = raw_det_result[0]
    # * filter boxes with small scores
    det_result = det_result[det_result[:, 4] >= args.det_score_thr]
    # * filter boxes with small areas
    box_areas = (det_result[:, 3] - det_result[:, 1]) * (det_result[:, 2] - det_result[:, 0])
    assert np.all(box_areas >= 0)
    det_result = det_result[box_areas >= args.det_area_thr]
    return det_result


def pose_inference_one_frame(model, frame, det_result):
    kp = np.zeros((1, 133, 3), dtype=np.float32)
    bb = np.zeros((1, 5), dtype=np.float32)
    d = [dict(bbox=x) for x in list(det_result)]
    pose = inference_top_down_pose_model(model, frame, d, format='xyxy')[0]
    pose = sorted(pose, key=lambda x:x['bbox'][-1])
    keypoints, bbox = pose[-1]['keypoints'], pose[-1]['bbox']
    kp[0] = keypoints
    bb[0] = bbox
    return kp


def inference_one_frame(args, det_model, pose_model, frame):
    det_result = detection_inference_one_frame(args, det_model, frame)
    kp= pose_inference_one_frame(pose_model, frame, det_result)
    return kp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.5)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='./tmp')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='2014T', choices=['2014', '2014T', 'csl-daily'])
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--img_per_iter', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '2'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'
    # num_gpus = torch.cuda.device_count()
    # rank = int(os.environ['RANK'])
    # print('hh')
    # dist.init_process_group(backend='nccl', rank=args.local_rank)
    # torch.cuda.set_device(args.local_rank)
    # print('hh')
    # rank, world_size = dist.get_rank(), dist.get_world_size()

    # if rank == 0:
    #     os.makedirs(args.tmpdir, exist_ok=True)
    # dist.barrier()
    len_dset, _ = create_dataloader(dset_name=args.dataset, split=args.split, bsize=1)
    # train_idx = np.arange(len_dset)[len_dset//2:]
    # spler = sampler.SequentialSampler(train_idx[:len_dset//2])

    _, valid_loader = create_dataloader(dset_name=args.dataset, split=args.split, bsize=1, sampler=None)

    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda')
    
    if args.dataset == '2014T':
        path = osp.join('/3tdisk/shared/rzuo/PHOENIX-2014-T', 'keypoints_hrnet_dark_coco_wholebody', args.split)
        sample_intv = 100
        h, w = 260, 210
    if args.dataset == '2014':
        path = osp.join('/3tdisk/shared/rzuo/PHOENIX-2014', 'keypoints_hrnet_dark_coco_wholebody', args.split)
        sample_intv = 999
        h, w = 260, 210
    elif args.dataset == 'csl-daily':
        path = osp.join('/3tdisk/shared/rzuo/CSL-Daily', 'keypoints_hrnet_dark_coco_wholebody')
        sample_intv = 200
        h, w = 512, 512
    if not osp.exists(path):
        os.makedirs(path)

    for k, batch_data in tqdm(enumerate(valid_loader), desc='[Generating keypoints of {:s} of {:s}]'.format(args.split, args.dataset)):

        len_video = batch_data['len_video'][0]
        video_id = batch_data['id'][0]
        fname = osp.join(path, ''.join(video_id))
        if ''.join(video_id)+'.npz' in os.listdir(path):
            data = np.load(fname+'.npz')
            assert data['keypoints'].shape == (len_video, 133, 3)
            continue

        frames = batch_data['video'][0].numpy().transpose(0,2,3,1)*255  #[T,H,W,3]
        frames = np.uint8(frames)
        frames = np.split(frames, frames.shape[0], axis=0)
        
        det_results = detection_inference(det_model, frames)
        # * Get detection results for human
        det_results = [x[0] for x in det_results]
        for i, res in enumerate(det_results):
            # * filter boxes with small scores
            res = res[res[:, 4] >= args.det_score_thr]
            # * filter boxes with small areas
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            det_results[i] = res

        pose_results = pose_inference(pose_model, frames, det_results)

        if k%sample_intv==0:
            # visulize video
            fps=15
            fourcc=cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter('vis_res/{:s}_{:s}.mp4'.format(args.dataset, ''.join(video_id)), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
            for idx in range(len(frames)):
                img = frames[idx][0, ..., ::-1].astype(np.uint8)
                cv2.imwrite('temp.jpg', img)
                img = cv2.imread('temp.jpg')
                # bb_x1, bb_y1, bb_x2, bb_y2 = bb_results[idx, :-1]
                # cv2.line(f, (int(bb_x1), int(bb_y1)), (int(bb_x1), int(bb_y2)), (0,255,0))
                # cv2.line(f, (int(bb_x1), int(bb_y1)), (int(bb_x2), int(bb_y1)), (0,255,0))
                # cv2.line(f, (int(bb_x2), int(bb_y2)), (int(bb_x2), int(bb_y1)), (0,255,0))
                # cv2.line(f, (int(bb_x2), int(bb_y2)), (int(bb_x1), int(bb_y2)), (0,255,0))
                # for j in range(133):
                #     x,y = pose_results[idx, j, :-1]
                #     x,y = int(x), int(y)
                #     cv2.circle(f, (x,y), 1, (0,0,255))
                # cv2.imwrite('temp.jpg', f)
                # f = cv2.imread('temp.jpg')
                # video_writer.write(f)

                vis_pose_result(pose_model,
                                img,
                                result=[{'keypoints': pose_results[idx]}],
                                radius=1,
                                thickness=1,
                                kpt_score_thr=0.3,
                                bbox_color='green',
                                dataset='TopDownCocoWholeBodyDataset',
                                dataset_info=None,
                                show=False,
                                out_file='temp.jpg')
                f = cv2.imread('temp.jpg')
                video_writer.write(f)
            video_writer.release()
        
        assert pose_results.shape == (len_video, 133, 3)
        np.savez_compressed(fname+'.npz', keypoints=pose_results.astype(np.float16))
    
    # dist.barrier()


if __name__ == '__main__':
    main()