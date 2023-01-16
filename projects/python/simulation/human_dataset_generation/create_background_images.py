# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# from pycocotools.coco import COCO
import numpy as np
# import matplotlib.pyplot as plt
from shutil import copyfile
import pickle
# import glob
import cv2
import argparse
import csv


# Keep only images from Cityscapes (a) without humans (persons/riders) (b) with roads/sidewalks/terrain
def add_cityscapes_background_imgs(rgb_in='./background_images/Cityscapes/in/all/rgb',
                                   segm_in='./background_images/Cityscapes/in/all/segm',
                                   human_colors='./background_images/Cityscapes/human_colormap.txt',
                                   placement_colors='./background_images/Cityscapes/locations_colormap.txt',
                                   imgs_dir_out='./background_images/out'):

    if not os.path.exists(os.path.join(imgs_dir_out, 'segm')):
        os.makedirs(os.path.join(imgs_dir_out, 'segm'))
    if not os.path.exists(os.path.join(imgs_dir_out, 'rgb')):
        os.makedirs(os.path.join(imgs_dir_out, 'rgb'))
    img_names = [f for f in os.listdir(segm_in) if os.path.isfile(os.path.join(segm_in, f))]

    # Pixel colors for humans (persons/riders)
    with open(human_colors) as csvfile:
        labels_rm = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row_ints = [int(i) for i in row]
            labels_rm.append(np.array(row_ints))

    # Pixel colors for roads/sidewalks/terrain
    with open(placement_colors) as csvfile:
        labels_pm = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row_ints = [int(i) for i in row]
            labels_pm.append(np.array(row_ints))

    for i in range(len(img_names)):

        rgb_path_in = os.path.join(rgb_in, img_names[i])
        segm_path_in = os.path.join(segm_in, img_names[i])
        rgb_path_out = os.path.join(imgs_dir_out, 'rgb', img_names[i])
        segm_path_out = os.path.join(imgs_dir_out, 'segm', img_names[i])
        img = cv2.imread(segm_path_in)
        msks_n = []
        for i in range(len(labels_rm)):
            msk_n = cv2.inRange(img, labels_rm[i] - 1, labels_rm[i] + 1)
            msks_n.append(msk_n)
        msk_n = msks_n[0]
        for i in range(len(labels_rm)):
            msk_n = cv2.bitwise_or(msk_n, msks_n[i])

        msks_p = []
        for i in range(len(labels_pm)):
            msk_p = cv2.inRange(img, labels_pm[i] - 1, labels_pm[i] + 1)
            msks_p.append(msk_p)
        msk_p = msks_p[0]
        for i in range(len(labels_pm)):
            msk_p = cv2.bitwise_or(msk_p, msks_p[i])

        if (not np.any(msk_n > 0)) and (np.sum(msk_p > 0) > 0.2 * msk_p.shape[0] * msk_p.shape[1]):
            copyfile(segm_path_in, segm_path_out)
            copyfile(rgb_path_in, rgb_path_out)


def generate_img_ids(imgs_dir='./background_images/out/rgb', imgs_dict_path='./background_images/img_ids.pkl', id_start=0):
    f = []
    for (dir_path, dirnames, filenames) in os.walk(imgs_dir):
        f.extend(filenames)
        break
    dict_ids = []
    for i in range(len(f)):
        dict_id = {
            'id': i + id_start,
            'filename': f[i]
        }
        dict_ids.append(dict_id)
    with open(imgs_dict_path, 'wb') as handle:
        pickle.dump(dict_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs_dict_path', type=str, default='./background_images/CityScapes/img_ids.pkl')
    parser.add_argument('-rgb_in', type=str, default='./background_images/CityScapes/in/all/rgb')
    parser.add_argument('-segm_in', type=str, default='./background_images/CityScapes/in/all/segm')
    parser.add_argument('-imgs_dir_out', type=str, default='./background_images/CityScapes/out')
    parser.add_argument('-human_colors', type=str, default='./background_images/CityScapes/human_colormap.txt')
    parser.add_argument('-placement_colors', type=str, default='./background_images/CityScapes/locations_colormap.txt')
    opt = parser.parse_args()

    # Reformate CitySCapes Dataset
    add_cityscapes_background_imgs(rgb_in=opt.rgb_in, segm_in=opt.segm_in,
                                   imgs_dir_out=opt.imgs_dir_out, human_colors=opt.human_colors,
                                   placement_colors=opt.placement_colors)
    '''
    generate_img_ids(imgs_dir_in=opt.imgs_dir_in, imgs_dict_path=opt.imgs_dict_path, id_start=1)
    '''
