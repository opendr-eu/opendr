#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import os
import math
from tqdm import tqdm
import time
# import face_alignment
# import single_align
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors, get_aligned_param, parse_quality_list_part
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.params import param_mean, param_std
from utils.render import get_depths_image, cget_depths_image, cpncc, crender_colors
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
from simple_dataset import McDataset
from torch.utils.data import DataLoader

STD_SIZE = 120


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # 2. parse images list and landmark
    lmk_file = args.lmk_file
    ts = time.time()
    rank_land, rank_img_list, start, end = parse_quality_list_part(lmk_file, args.world_size, args.rank, args.resume_idx)
    print('parse land file in {:.3f} seconds'.format(time.time() - ts))

    # for batch processing
    print('World size {}, rank {}, start from {}, end with {}'.format(args.world_size, args.rank, start, end))
    dataset = McDataset(rank_img_list, rank_land, transform=transform, std_size=STD_SIZE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for img_idx, (inputs, ori_imgs, img_fps, roi_boxes) in enumerate(tqdm(dataloader)):

        # forward: one step
        with torch.no_grad():
            if args.mode == 'gpu':
                inputs = inputs.cuda()
            params = model(inputs)
            params = params.cpu().numpy()

        roi_boxes = roi_boxes.numpy()
        outputs_roi_boxes = roi_boxes
        if args.bbox_init == 'two':
            step_two_ori_imgs = []
            step_two_roi_boxes = []
            ori_imgs = ori_imgs.numpy()
            for ii in range(params.shape[0]):
                # 68 pts
                pts68 = predict_68pts(params[ii], roi_boxes[ii])

                # two-step for more accurate bbox to crop face
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(ori_imgs[ii], roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                # input = transform(img_step2).unsqueeze(0)
                step_two_ori_imgs.append(transform(img_step2))
                step_two_roi_boxes.append(roi_box)
            with torch.no_grad():
                step_two_ori_imgs = torch.stack(step_two_ori_imgs, dim=0)
                inputs = step_two_ori_imgs
                if args.mode == 'gpu':
                    inputs = inputs.cuda()
                params = model(inputs)
                params = params.cpu().numpy()
            outputs_roi_boxes = step_two_roi_boxes

        # dump results
        if args.dump_param:
            for img_fp, param, roi_box in zip(img_fps, params, outputs_roi_boxes):
                split = img_fp.split('/')
                save_name = os.path.join(args.save_dir, '{}.txt'.format(os.path.splitext(split[-1])[0]))
                this_param = param * param_std + param_mean
                this_param = np.concatenate((this_param, roi_box))
                this_param.tofile(save_name, sep=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--bbox_init', default='two', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_2d_img', default='false', type=str2bool, help='whether to save 3d rendered image')
    parser.add_argument('--dump_param', default='true', type=str2bool, help='whether to save param')
    parser.add_argument('--save_dir', default='results', type=str, help='dir to save result')
    parser.add_argument('--lmk_file', default='quality_list', type=str, help='landmarks file')
    parser.add_argument('--rank', default=0, type=int, help='used when parallel run')
    parser.add_argument('--world_size', default=1, type=int, help='used when parallel run')
    parser.add_argument('--resume_idx', default=0, type=int)
    parser.add_argument('--batch_size', default=80, type=int, help='batch size')

    args = parser.parse_args()
    main(args)

