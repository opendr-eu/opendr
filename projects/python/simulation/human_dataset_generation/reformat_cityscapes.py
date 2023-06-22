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
import argparse
import glob
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default='./background_images/Cityscapes/in')
    opt = parser.parse_args()

    all_dir = os.path.join(opt.data_dir, 'all')
    if not os.path.isdir(all_dir):
        os.mkdir(all_dir)

    segm_dir = os.path.join(opt.data_dir, 'all', 'segm')
    if not os.path.isdir(segm_dir):
        os.mkdir(segm_dir)

    gtCoarse = os.path.join(opt.data_dir, 'gtCoarse''*', '*', '*', '*')
    for segm_img in glob.glob(gtCoarse):
        if '_gtCoarse_color.png' in segm_img:
            segm_img_new = os.path.join(segm_dir, os.path.basename(segm_img).replace('_gtCoarse_color', ''))
            shutil.move(segm_img, segm_img_new)
    # shutil.rmtree(os.path.join(opt.data_dir,'gtCoarse'))

    rgb_dir = os.path.join(opt.data_dir, 'all', 'rgb')
    if not os.path.isdir(rgb_dir):
        os.mkdir(rgb_dir)

    leftImg8bit = os.path.join(opt.data_dir, 'leftImg8bit''*', '*', '*', '*')
    for rgb_img in glob.glob(leftImg8bit):
        if '_leftImg8bit.png' in rgb_img:
            rgb_img_new = os.path.join(rgb_dir, os.path.basename(rgb_img).replace('_leftImg8bit', ''))
            shutil.move(rgb_img, rgb_img_new)
    # shutil.rmtree(os.path.join(opt.data_dir,'leftImg8bit'))
