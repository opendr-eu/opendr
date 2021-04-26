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
import numpy as np
import math
from typing import Callable, Tuple
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
    ToTensorVideo,
)

Transform = Callable[[torch.Tensor], torch.Tensor]


def standard_video_transforms(
    spatial_pixels: int = 224,
    horizontal_flip=True,
    mean=(0.45, 0.45, 0.45),
    std=(0.225, 0.225, 0.225)
) -> Tuple[Transform, Transform]:
    """Generate standard transforms for video recognition

    Args:
        spatial_pixels (int, optional ): Spatial size (i.e. height or width) to resize to. Defaults to 224.
        horizontal_flip (bool, optional): Whether horizontal flipping (p = 0.5) is used. Defaults to True.
        mean (tuple, optional): Mean RGB values used in standardization. Defaults to (0.45, 0.45, 0.45).
        std (tuple, optional): Std RGB values used in standardization. Defaults to (0.225, 0.225, 0.225).

    Returns:
        Tuple[Transform, Transform]: [description]
    """
    train_scale = (1 / 0.7 * 0.8, 1 / 0.7)
    scaled_pix_min = (spatial_pixels * train_scale[0]) // 2 * 2
    scaled_pix_max = (spatial_pixels * train_scale[1]) // 2 * 2
    train_transforms = Compose(
        [
            t for t in [
                ToTensorVideo(),
                RandomShortSideScaleJitterVideo(
                    min_size=scaled_pix_min, max_size=scaled_pix_max
                ),
                RandomCropVideo(spatial_pixels),
                RandomHorizontalFlipVideo() if horizontal_flip else None,
                NormalizeVideo(mean=mean, std=std),
            ] if t
        ]
    )
    eval_transforms = Compose(
        [
            ToTensorVideo(),
            RandomShortSideScaleJitterVideo(min_size=spatial_pixels, max_size=spatial_pixels),
            CenterCropVideo(spatial_pixels),
            NormalizeVideo(mean=mean, std=std),
        ]
    )
    return train_transforms, eval_transforms


class RandomShortSideScaleJitterVideo:
    def __init__(self, min_size: int, max_size: int, inverse_uniform_sampling=False):
        """
        Args:
            min_size (int): the minimal size to scale the frames.
            max_size (int): the maximal size to scale the frames.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale, max_scale].
        """
        self.min_size = min_size
        self.max_size = max_size
        self.inverse_uniform_sampling = inverse_uniform_sampling

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perform a spatial short scale jittering on the given images and
        corresponding boxes.
        Args:
            images (tensor): images to perform scale jitter. Dimension is
                `num frames` x `channel` x `height` x `width`.
        Returns:
            (tensor): the scaled images with dimension of
                `num frames` x `channel` x `new height` x `new width`.
        """
        if self.inverse_uniform_sampling:
            size = int(
                round(1.0 / np.random.uniform(1.0 / self.max_size, 1.0 / self.min_size))
            )
        else:
            size = int(round(np.random.uniform(self.min_size, self.max_size)))

        height = images.shape[2]
        width = images.shape[3]
        if (width <= height and width == size) or (height <= width and height == size):
            return images
        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        return torch.nn.functional.interpolate(
            images, size=(new_height, new_width), mode="bilinear", align_corners=False,
        )
