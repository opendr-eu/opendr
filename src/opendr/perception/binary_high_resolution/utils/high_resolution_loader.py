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


import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import glob
import cv2
import numpy as np
from skimage.util import view_as_blocks
from tqdm import tqdm
import PIL


def parse_annotation_file(path):
    bboxes = []
    labels = []
    tree = ET.parse(path)
    root = tree.getroot()
    filename = root.find('filename').text
    for obj in root.iter('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
        labels.append(name)

    return bboxes, labels, filename


class HighResolutionDataset(Dataset):

    def __init__(self, dataset_folder, patch_size=64, non_zero_threshold=0.5, transform=None):
        """
        Loads the images (expected to be in VOC2012 format) and splits them into patches of size patch_size
        :param dataset_folder: folder from which the data are loaded
        :param patch_size: size of the patches to extract from images
        :param non_zero_threshold: threshold for discarding a patch (if a patch contains zeros more that this threshold
        is discard)
        :param transform: PyTorch transformation to be applied on each patch
        """
        self.patches = []
        self.labels = []
        self.transform = transform

        xml_files = glob.glob(dataset_folder + '/*.xml')

        print("Processing input images...")
        for file in tqdm(xml_files):
            cur_bboxes, cur_labels, filename = parse_annotation_file(file)
            path = file[:file.rfind('/') + 1] + filename
            cur_img = cv2.imread(path)

            for i in range(len(cur_bboxes)):
                (xmin, ymin, xmax, ymax) = cur_bboxes[i]
                if xmax - xmin < patch_size:
                    diff = patch_size - (xmax - xmin)
                    xmax += int((diff + 1) / 2)
                    xmin -= int(diff / 2)
                if ymax - ymin < patch_size:
                    diff = patch_size - (ymax - ymin)
                    ymax += int((diff + 1) / 2)
                    ymin -= int(diff / 2)

                pos_patch = np.copy(cur_img[ymin:ymax, xmin:xmax])
                # Pad the cropped area
                height, width = pos_patch.shape[:2]
                h_pad = patch_size - height % patch_size
                w_pad = patch_size - width % patch_size
                pad = ((0, h_pad), (0, w_pad), (0, 0))
                pos_patch = np.pad(pos_patch, pad, mode='constant')

                # Get positive patches
                cur_patches = view_as_blocks(pos_patch, (patch_size, patch_size, 3))
                cur_patches = cur_patches.reshape((-1, patch_size, patch_size, 3))
                if type(self.labels) != list:
                    self.labels = self.labels.tolist()
                self.patches.extend(cur_patches)
                self.labels.extend(np.ones((cur_patches.shape[0], 1)))

            # Zero the corresponding area in the original image
            for i in range(len(cur_bboxes)):
                (xmin, ymin, xmax, ymax) = cur_bboxes[i]
                pos_patch = np.copy(cur_img[ymin:ymax, xmin:xmax])
                cur_img[ymin:ymax, xmin:xmax] = 0

            # Pad the cropped area
            height, width = cur_img.shape[:2]
            h_pad = patch_size - height % patch_size
            w_pad = patch_size - width % patch_size
            pad = ((0, h_pad), (0, w_pad), (0, 0))
            cur_img = np.pad(cur_img, pad, mode='constant')
            # Add negative patches
            neg_patches = view_as_blocks(cur_img, (patch_size, patch_size, 3))
            neg_patches = neg_patches.reshape((-1, patch_size, patch_size, 3))
            self.patches.extend(neg_patches)
            self.labels.extend(np.zeros((neg_patches.shape[0], 1)))

            self.patches = np.asarray(self.patches)
            self.labels = np.asarray(self.labels)

            # Remove patches that are almost zero
            idx = np.ones((len(self.labels, )), dtype=np.bool_)
            for i in range(len(self.labels)):
                if np.sum(self.patches[i] == 0) > non_zero_threshold * patch_size * patch_size * 3:
                    idx[i] = False
            patches = self.patches[idx]
            labels = self.labels[idx]

            self.patches = []
            self.labels = labels
            # Convert to PIL images to be used with PyTorch
            for i in range(len(labels)):
                img = patches[i]
                try:
                    self.patches.append(PIL.Image.fromarray(img))
                except TypeError:
                    self.patches.append(PIL.Image.fromarray(np.array(img)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.patches[idx]
        label = torch.tensor(np.int64(self.labels[idx]))
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        return np.asarray([x.item() for x in self.labels])
