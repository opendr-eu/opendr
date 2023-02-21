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

"""
This example shows one way to convert a DetectionDataset, namely the AGIHumans dataset, to YOLOv5 format,
to be used for training a custom model using the original YOLOv5 implementation: https://github.com/ultralytics/yolov5
The opendr_datasets package can be installed using: `pip install git+https://github.com/opendr-eu/datasets.git`
"""

import argparse
import os
import yaml
import cv2
from opendr_datasets import AGIHumans


def main(args):
    path = args.original_data_path
    train_set = AGIHumans(path, train=True)
    val_set = AGIHumans(path, train=False)

    new_path = args.new_data_path
    os.makedirs(new_path, exist_ok=True)

    # step 1: write dataset .yml file
    # the new data structure is as follows:
    # new_path
    # ├── train
    # │  ├── images
    # │  │  ├── im00001.jpg
    # │  │  └── ...
    # │  └── labels
    # │     ├── im00001.txt
    # │     └── ...
    # ├── test
    # │  ├── images
    # │  │  ├── im00001.jpg
    # │  │  └── ...
    # │  └── labels
    # │     ├── im00001.txt
    # │     └── ...
    # └── AGIHumans.yml

    d = {
        'path': new_path,
        'train': 'train',
        'val': 'test',
        'names': {c: c_name for c, c_name in enumerate(train_set.class_names)}
    }

    with open('AGIHumans.yaml', 'w') as yaml_file:
        yaml.dump(d, yaml_file, default_flow_style=False)

    # step 2: convert annotations to .txt files
    # train set
    os.makedirs(os.path.join(new_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_path, 'train', 'labels'), exist_ok=True)
    for idx, (img, boxes) in enumerate(train_set):
        # save img to 'train/images/im{:05d}.jpg'
        im_cv = img.opencv()
        cv2.imwrite(os.path.join(new_path, 'train', 'images', f'im{idx:05d}.jpg'), im_cv)
        im_height, im_width, im_c = im_cv.shape
        # save normalized label to 'train/labels/im{:05d}.txt
        lines = ''
        for box in boxes:
            x_center = (box.left + box.width * 0.5) / im_width
            y_center = (box.top + box.height * 0.5) / im_height
            width = box.width / im_width
            height = box.height / im_height
            lines += f'{box.name} {x_center} {y_center} {width} {height}\n'
        if len(lines) > 0:
            with open(os.path.join(new_path, 'train', 'labels', f'im{idx:05d}.txt'), 'w') as f:
                f.write(lines)

    # validation/test set
    os.makedirs(os.path.join(new_path, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_path, 'test', 'labels'), exist_ok=True)
    for idx, (img, boxes) in enumerate(val_set):
        # save img to 'train/images/im{:05d}.jpg'
        im_cv = img.opencv()
        cv2.imwrite(os.path.join(new_path, 'test', 'images', f'im{idx:05d}.jpg'), im_cv)
        im_height, im_width, im_c = im_cv.shape
        # save normalized label to 'train/labels/im{:05d}.txt
        lines = ''
        for box in boxes:
            x_center = (box.left + box.width * 0.5) / im_width
            y_center = (box.top + box.height * 0.5) / im_height
            width = box.width / im_width
            height = box.height / im_height
            lines += f'{box.name} {x_center} {y_center} {width} {height}\n'
        if len(lines) > 0:
            with open(os.path.join(new_path, 'test', 'labels', f'im{idx:05d}.txt'), 'w') as f:
                f.write(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-data-path", help="Dataset root", type=str)
    parser.add_argument("--new-data-path", help="Path to converted dataset location", type=str)

    args = parser.parse_args()
