# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Modifications Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.utils.data
import torchvision


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(img_folder, ann_folder, ann_file, image_set, return_masks, dataset_type):
    if dataset_type == 'coco':
        from .coco import build as build_coco
        return build_coco(img_folder, ann_file, image_set, return_masks)
    if dataset_type == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(img_folder, ann_folder, ann_file, image_set, return_masks)
    raise ValueError(f'dataset {dataset_type} not supported')
