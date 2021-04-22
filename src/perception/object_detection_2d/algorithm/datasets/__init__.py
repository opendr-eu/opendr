# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


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
        return build_coco(img_folder, ann_file, image_set, return_masks)
    if dataset_type == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(img_folder, ann_folder, ann_file, image_set, return_masks)
    raise ValueError(f'dataset {dataset_type} not supported')
