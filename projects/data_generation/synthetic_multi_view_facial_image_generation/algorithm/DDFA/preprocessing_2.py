#!/usr/bin/env python3
# coding: utf-8

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
from . import mobilenet_v1
import numpy as np
import cv2
import os
from tqdm import tqdm
import face_alignment
from .utils.ddfa import ToTensorGjz, NormalizeGjz
import scipy.io as sio
from .utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts, predict_dense, get_colors, \
    get_5lmk_from_68lmk
from .utils.estimate_pose import parse_pose
from .utils.params import param_mean, param_std
from .utils.render import crender_colors
import torch.backends.cudnn as cudnn
__author__ = 'cleardusk'
STD_SIZE = 120


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'algorithm/DDFA/models/phase1_wpdc_vdc.pth.tar'
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

    tri = sio.loadmat('algorithm/DDFA/visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    # 2. parse images list
    with open(args.img_list) as f:
        img_list = [x.strip() for x in f.readlines()]
    landmark_list = []

    alignment_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.save_lmk_dir):
        os.mkdir(args.save_lmk_dir)

    for img_idx, img_fp in enumerate(tqdm(img_list)):
        img_ori = cv2.imread(os.path.join(args.img_prefix, img_fp))
        print("Image", img_fp)
        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        # vertices_lst = []  # store multiple face vertices
        # ind = 0
        # suffix = get_suffix(img_fp)

        # face alignment model use RGB as input, result is a tuple with landmarks and boxes
        preds = alignment_model.get_landmarks(img_ori[:, :, ::-1])
        pts_2d_68 = preds[0]
        pts_2d_5 = get_5lmk_from_68lmk(pts_2d_68)
        landmark_list.append(pts_2d_5)
        roi_box = parse_roi_box_from_landmark(pts_2d_68.T)

        img = crop_img(img_ori, roi_box)
        # import pdb; pdb.set_trace()

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            if args.mode == 'gpu':
                input = input.cuda()
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        # two-step for more accurate bbox to crop face
        if args.bbox_init == 'two':
            roi_box = parse_roi_box_from_landmark(pts68)
            img_step2 = crop_img(img_ori, roi_box)
            img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img_step2).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, roi_box)

        pts_res.append(pts68)
        P, pose = parse_pose(param)
        Ps.append(P)
        poses.append(pose)

        # dense face 3d vertices
        vertices = predict_dense(param, roi_box)

        if args.dump_2d_img:
            wfp_2d_img = os.path.join(args.save_dir, os.path.basename(img_fp))
            colors = get_colors(img_ori, vertices)
            # aligned_param = get_aligned_param(param)
            # vertices_aligned = predict_dense(aligned_param, roi_box)
            # h, w, c = 120, 120, 3
            h, w, c = img_ori.shape
            img_2d = crender_colors(vertices.T, (tri - 1).T, colors[:, ::-1], h, w)
            cv2.imwrite(wfp_2d_img, img_2d[:, :, ::-1])
        if args.dump_param:
            split = img_fp.split('/')
            save_name = os.path.join(args.save_dir, '{}.txt'.format(os.path.splitext(split[-1])[0]))
            this_param = param * param_std + param_mean
            this_param = np.concatenate((this_param, roi_box))
            this_param.tofile(save_name, sep=' ')
    if args.dump_lmk:
        save_path = os.path.join(args.save_lmk_dir, 'realign_lmk_')
        with open(save_path, 'w') as f:
            for idx, (fname, land) in enumerate(zip(img_list, landmark_list)):
                # f.write('{} {} {} {}')
                land = land.astype(np.int)
                land_str = ' '.join([str(x) for x in land])
                msg = f'{fname} {idx} {land_str}\n'
                f.write(msg)


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--bbox_init', default='two', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_2d_img', default='true', type=str2bool, help='whether to save 3d rendered image')
    parser.add_argument('--dump_param', default='true', type=str2bool, help='whether to save param')
    parser.add_argument('--dump_lmk', default='true', type=str2bool, help='whether to save landmarks')
    parser.add_argument('--save_dir', default='results', type=str, help='dir to save result')
    parser.add_argument('--save_lmk_dir', default='example', type=str, help='dir to save landmark result')
    parser.add_argument('--img_list', default='example/file_list.txt', type=str, help='test image list file')
    parser.add_argument('--img_prefix', default='example/Images/', type=str, help='test image prefix')
    parser.add_argument('--rank', default=0, type=int, help='used when parallel run')
    parser.add_argument('--world_size', default=1, type=int, help='used when parallel run')
    parser.add_argument('--resume_idx', default=0, type=int)

    args = parser.parse_args()
    '''
    # main(args)
