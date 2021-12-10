# Copyright 2020-2021 OpenDR European Project
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

import torch
import numpy as np
import imageio
import os
from opendr.engine.datasets import DatasetIterator
from urllib.request import urlretrieve
import zipfile
import torchvision.transforms as transforms


class RgbdDataset(DatasetIterator):
    def __init__(self, annotation, transforms=None):
        self._verify_annotation(annotation)
        self.annotation = annotation
        self.transforms = transforms

    def __len__(self,):
        return len(self.annotation)

    def _verify_annotation(self, annotation):
        for row in annotation:
            assert len(row) == 4,\
                'Each element in annotation list must be a 4-element tuple/list ' +\
                'that contains rgb path, depth path, text label, int label'
            rgb_file, depth_file, _, _ = row
            assert os.path.exists(rgb_file),\
                '{} does not exist'.format(rgb_file)
            assert os.path.exists(depth_file),\
                '{} does not exist'.format(depth_file)

    def __getitem__(self, i):
        rgb_file, depth_file, text_label, int_label = self.annotation[i]
        rgb = np.asarray(imageio.imread(rgb_file)) / 255.0
        depth = np.asarray(imageio.imread(depth_file)) / 65535.0
        depth = np.expand_dims(depth, axis=-1)
        img = np.concatenate([rgb, depth], axis=-1).astype('float32')

        if self.transforms is not None:
            img = self.transforms(img)

        return img.float(), torch.tensor([int_label, ]).long()


class DataWrapper:
    def __init__(self, opendr_dataset):
        self.dataset = opendr_dataset

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset.__getitem__(i)
        # change from rows x cols x channels to channels x rows x cols
        x = x.convert("channels_first")
        return torch.from_numpy(x).float(), torch.tensor([y.data, ]).long()


def get_annotation(src):
    train_folders = ['Subject1', 'Subject2', 'Subject3']
    test_folders = ['Subject4', 'Subject5']

    empty_list = '[0 0 0 0]'

    train_labels = []
    for folder in train_folders:
        sub_dir = os.path.join(src, folder)
        label_file = os.path.join(sub_dir, folder + '.txt')

        fid = open(label_file, 'r')
        content = fid.read().split('\n')[:-1]
        fid.close()

        text_lb = content[0].split(',')[2:]

        for row in content[1:]:
            parts = row.split(',')
            rgb_file = parts[0].split('\\')[-1]
            rgb_file = os.path.join(sub_dir, folder, rgb_file)
            depth_file = parts[1].split('\\')[-1].replace('color', 'depth')
            depth_file = os.path.join(sub_dir, folder + '_Depth', depth_file)
            lb = [rgb_file, depth_file]
            for idx, p in enumerate(parts[2:]):
                if p != empty_list:
                    lb.append(text_lb[idx])
            train_labels.append(lb)

    test_labels = []
    for folder in test_folders:
        sub_dir = os.path.join(src, folder)
        label_file = os.path.join(sub_dir, folder + '.txt')
        fid = open(label_file, 'r')
        content = fid.read().split('\n')[:-1]
        fid.close()

        text_lb = content[0].split(',')[2:]

        for row in content[1:]:
            parts = row.split(',')
            rgb_file = parts[0].split('\\')[-1]
            rgb_file = os.path.join(sub_dir, folder, rgb_file)
            depth_file = parts[1].split('\\')[-1]
            depth_file = os.path.join(sub_dir, folder + '_Depth', depth_file)
            lb = [rgb_file, depth_file]
            for idx, p in enumerate(parts[2:]):
                if p != empty_list:
                    lb.append(text_lb[idx])
            test_labels.append(lb)

    train_labels, text_labels = refine_label(train_labels)
    test_labels = refine_label(test_labels)[0]

    return train_labels, test_labels, len(text_labels), text_labels


def refine_label(labels):
    text_labels = set()
    for lb in labels:
        text_labels.add('_AND_'.join(lb[2:]))
    text_labels = list(text_labels)
    text_labels.sort()
    new_labels = []
    for lb in labels:
        text = '_AND_'.join(lb[2:])
        idx = text_labels.index(text)
        new_labels.append((lb[0], lb[1], text, idx))

    return new_labels, text_labels


def get_hand_gesture_dataset(path, resolution=224):
    src = os.path.join(path, 'hand_gestures')
    if not os.path.exists(src):
        # if data not available, download
        url = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/ndrczc35bt-2.zip'
        zip_file = os.path.join(path, 'data.zip')
        urlretrieve(url, zip_file)
        with zipfile.ZipFile(zip_file, 'r') as fid:
            fid.extractall(src)
        os.remove(zip_file)
        # extract zip file in each individual directories
        for i in range(1, 6):
            sub_src = os.path.join(src, 'Subject{}'.format(i))
            rgb_zip_file = os.path.join(sub_src, 'Subject{}.zip'.format(i))
            depth_zip_file = os.path.join(sub_src, 'Subject{}_Depth.zip'.format(i))
            with zipfile.ZipFile(rgb_zip_file, 'r') as fid:
                fid.extractall(sub_src)
            with zipfile.ZipFile(depth_zip_file, 'r') as fid:
                fid.extractall(sub_src)
            os.remove(rgb_zip_file)
            os.remove(depth_zip_file)

    train_labels, val_labels, n_class, text_labels = get_annotation(src)

    mean = [0.485, 0.456, 0.406, 0.0303]
    std = [0.229, 0.224, 0.225, 0.0353]

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(resolution, scale=(0.8, 1.0)),
        transforms.Normalize(mean, std)
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resolution),
        transforms.Normalize(mean, std)
    ])

    train_set = RgbdDataset(train_labels, train_transforms)
    val_set = RgbdDataset(val_labels, val_transforms)

    return train_set, val_set, n_class, text_labels
