# Copyright 2020 Aristotle University of Thessaloniki
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
import numpy as np
from shutil import copyfile
import pickle5 as pickle
import glob
import cv2
import argparse


def add_cityscapes_background_imgs(imgs_dir_in='./background_images/in/CityScapes',
                                   imgs_dir_out='./background_images/out'):
    """
    Keep only images from CityScapes (a) without humans (persons/riders) (b) with roads/sidewalks/terrain
    """
    if not os.path.exists(os.path.join(imgs_dir_out, 'labels')):
        os.makedirs(os.path.join(imgs_dir_out, 'labels'))
    if not os.path.exists(os.path.join(imgs_dir_out, 'rgb')):
        os.makedirs(os.path.join(imgs_dir_out, 'rgb'))
    segm_imgs = glob.glob(imgs_dir_in + '/gtCoarse/*/*/**gtCoarse_color.png', recursive=True)
    # Pixel colors for humans (persons/riders)
    labels_rm = [np.array([60, 20, 220]), np.array([255, 0, 0])]
    # Pixel colors for roads/sidewalks/terrain
    labels_pm = [np.array([128, 64, 128]), np.array([232, 35, 244]), np.array([152, 251, 152])]

    for i in range(len(segm_imgs)):
        segm_filename = os.path.basename(segm_imgs[i])
        rgb_filename = segm_filename.replace('gtCoarse_color', 'leftImg8bit')
        out_filename = segm_filename.replace('_gtCoarse_color', '')
        segm_path_in = segm_imgs[i]

        segm_path_out = os.path.join(imgs_dir_out, 'labels', out_filename)
        rgb_path_in = glob.glob(imgs_dir_in + '/leftImg8bit/*/*/' + rgb_filename)[0]
        rgb_path_out = os.path.join(imgs_dir_out, 'rgb', out_filename)
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


def generate_img_ids(imgs_dir='./background_images/out/rgb', imgs_dict_path='./background_images/img_ids.pkl',
                     id_start=0):
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
    parser.add_argument('-imgs_dict_path', type=str, default='./background_images/out/img_ids.pkl')
    parser.add_argument('-imgs_dir_in', type=str, default='./background_images/in/CityScapes')
    parser.add_argument('-imgs_dir_out', type=str, default='./background_images/out')
    opt = parser.parse_args()

    # Reformate CitySCapes Dataset
    add_cityscapes_background_imgs(imgs_dir_in=opt.imgs_dir_in,
                                   imgs_dir_out=opt.imgs_dir_out)
    '''
    generate_img_ids(imgs_dir_in=opt.imgs_dir_in, imgs_dict_path=opt.imgs_dict_path, id_start=1)
    '''
