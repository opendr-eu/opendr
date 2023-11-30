# Copyright 2021 RangiLyu.
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
import random
import psutil

from abc import ABCMeta, abstractmethod
from typing import Tuple
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.transform import Pipeline
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util import get_size, mkdir


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    A base class of detection dataset. Referring from MMDetection.
    A dataset should have images, annotations and preprocessing pipelines
    NanoDet use [xmin, ymin, xmax, ymax] format for box and
     [[x0,y0], [x1,y1] ... [xn,yn]] format for key points.
    instance masks should decode into binary masks for each instance like
    {
        'bbox': [xmin,ymin,xmax,ymax],
        'mask': mask
     }
    segmentation mask should decode into binary masks for each class.
    Args:
        img_path (str): image data folder
        ann_path (str): annotation file path or folder
        use_instance_mask (bool): load instance segmentation data
        use_seg_mask (bool): load semantic segmentation data
        use_keypoint (bool): load pose keypoint data
        load_mosaic (bool): using mosaic data augmentation from yolov4
        mode (str): 'train' or 'val' or 'test'
        multi_scale (Tuple[float, float]): Multi-scale factor range.
    """

    def __init__(
        self,
        img_path,
        ann_path,
        input_size,
        pipeline,
        keep_ratio=True,
        use_instance_mask=False,
        use_seg_mask=False,
        use_keypoint=False,
        load_mosaic=False,
        mode="train",
        multi_scale=None,
        cache_images="_"
    ):
        assert mode in ["train", "val", "test"]
        self.img_path = img_path
        self.ann_path = ann_path
        self.input_size = input_size
        self.pipeline = Pipeline(pipeline, keep_ratio)
        self.keep_ratio = keep_ratio
        self.use_instance_mask = use_instance_mask
        self.use_seg_mask = use_seg_mask
        self.use_keypoint = use_keypoint
        self.load_mosaic = load_mosaic
        self.multi_scale = multi_scale
        self.mode = mode

        print(ann_path)
        self.data_info = self.get_data_info(ann_path)

        # Cache images into RAM/disk for faster training
        self.metas = [{}] * len(self)
        self.npy_files = [None] * len(self)
        cache_images = None if cache_images == "_" else cache_images
        if cache_images == 'ram' and not self.check_cache_ram(prefix=mode):
            cache_images = False

        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.get_data
            results = ThreadPool(NUM_THREADS).imap(fcn, range(len(self)))
            pbar = tqdm(enumerate(results), total=len(self), bar_format=TQDM_BAR_FORMAT)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.metas[i] = x  # meta = dict(img, height, width, id, file_name, gt_bboxes, gt_labels)
                    b += get_size(self.metas[i])
                pbar.desc = f'{mode}: Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()
        return

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if self.mode in ["val", "test"]:
            return self.get_val_data(idx)
        else:
            while True:
                data = self.get_train_data(idx)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    def __call__(self, idx):
        if self.mode in ["val", "test"]:
            return self.get_val_data(idx)
        else:
            while True:
                data = self.get_train_data(idx)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    @staticmethod
    def get_random_size(
        scale_range: Tuple[float, float], image_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get random image shape by multi-scale factor and image_size.
        Args:
            scale_range (Tuple[float, float]): Multi-scale factor range.
                Format in [(width, height), (width, height)]
            image_size (Tuple[int, int]): Image size. Format in (width, height).

        Returns:
            Tuple[int, int]
        """
        assert len(scale_range) == 2
        scale_factor = random.uniform(*scale_range)
        width = int(image_size[0] * scale_factor)
        height = int(image_size[1] * scale_factor)
        return width, height

    @abstractmethod
    def get_data_info(self, ann_path):
        pass

    @abstractmethod
    def get_train_data(self, idx):
        pass

    @abstractmethod
    def get_data(self, idx):
        pass

    @abstractmethod
    def get_per_img_info(self, idx):
        pass

    @abstractmethod
    def get_val_data(self, idx):
        pass

    def get_another_id(self):
        return np.random.random_integers(0, len(self.data_info) - 1)

    def check_cache_ram(self, safety_margin=0.1, prefix=''):
        # Check image caching requirements vs available memory
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(len(self), 30)  # extrapolate from 30 random images
        for _ in range(n):
            meta = self.get_train_data(random.choice(range(len(self))))
            b += get_size(meta)
        mem_required = b * len(self) / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            print(f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                  f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                  f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        meta = self.get_per_img_info(i)
        f = Path(os.path.join(self.img_path, "npys", meta["file_name"])).with_suffix(".npy")
        if not f.exists():
            mkdir(-1, os.path.join(self.img_path, "npys"), exist_ok=True)
            meta = self.get_data(i)
            np.save(f.as_posix(), meta["img"])
        self.npy_files[i] = f
