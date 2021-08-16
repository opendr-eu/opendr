# Copyright 2020-2021 OpenDR Project
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
import glob
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import random
import zipfile
from urllib.request import urlretrieve
from opendr.perception.semantic_segmentation.bisenet.algorithm.utils import get_label_info, RandomCrop, one_hot_it_v11
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from opendr.engine.constants import OPENDR_SERVER_URL


class CamVidDataset(ExternalDataset, DatasetIterator, torch.utils.data.Dataset):
    """
    CamVid dataset for Semantic Segmentation
    path (dir): path directory of the CamVid dataset, containinig the subfolders "train",
     "train_labels", "test", "test_labels", and the class_dict.csv file
    scale (int): (crop_height, crop_width)
    mode: 'test', 'train'
    """

    def __init__(self,
                 path: str,
                 scale=(720, 960),
                 mode='train'):
        ExternalDataset.__init__(self, path=str(path), dataset_type="camvid")
        DatasetIterator.__init__(self)
        torch.utils.data.Dataset.__init__(self)

        self.mode = mode
        if self.mode == 'train':
            self.image_path = os.path.join(self.path, 'train')
            self.label_path = os.path.join(self.path, 'train_labels')
        elif self.mode == 'test':
            self.image_path = os.path.join(self.path, 'test')
            self.label_path = os.path.join(self.path, 'test_labels')
        self.image_list = []
        if not isinstance(self.image_path, list):
            self.image_path = [self.image_path]
        for image_path_ in self.image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()
        self.label_list = []
        if not isinstance(self.label_path, list):
            self.label_path = [self.label_path]
        for label_path_ in self.label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
        self.fliplr = iaa.Fliplr(0.5)
        self.label_info = get_label_info(os.path.join(path, "class_dict.csv"))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # =====================================
        img = np.array(img)
        # load label
        label = Image.open(self.label_list[index])
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================
        label = np.array(label)
        # augment image and label
        if self.mode == 'train':
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            label = seq_det.augment_image(label)
        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()
        label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
        label = torch.from_numpy(label).long()

        return img, label

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def download_data(path: str):
        """Download camvid dataset

        Args:
            path: Directory in which to store the dataset
        """
        os.makedirs(path)
        url = os.path.join(
            OPENDR_SERVER_URL,
            "perception",
            "semantic_segmentation",
            "bisenet",
            "datasets",
            "CamVid.zip"
        )
        zip_path = os.path.join(path, 'CamVid.zip')
        urlretrieve(url=url, filename=zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        os.remove(zip_path)
