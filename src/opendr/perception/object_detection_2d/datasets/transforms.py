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

import cv2
import numpy as np
import mxnet as mx
import gluoncv.data.transforms.image as timage


def np_to_mx(img_np):
    """
    Convert numpy image to MXNet image.
    """
    img_mx = mx.image.image.nd.from_numpy(np.float32(img_np))
    return img_mx


def bbox_to_np(bbox):
    """
    BoundingBox to [xmin, ymin, xmax, ymax, conf, cls] numpy array.
    """
    bbox_np = np.asarray([bbox.left, bbox.top, bbox.left + bbox.width, bbox.top + bbox.height, bbox.confidence, bbox.name])
    return bbox_np


class BoundingBoxListToNumpyArray:
    """
    Transform object to convert OpenDR BoundingBoxList to numpy array of [[xmin, ymin, xmax, ymax, score, cls_id],...] format.
    """
    def __call__(self, bbox_list):
        return np.asarray([bbox_to_np(bbox) for bbox in bbox_list.data])


class ImageToNDArrayTransform:
    """
    Transform object to convert OpenDR Image to MXNext image.
    """
    def __call__(self, img):
        return np_to_mx(img.data)


class ImageToNumpyArrayTransform:
    """
    Transform object to convert OpenDR Image to Numpy array.
    """
    def __call__(self, img):
        return img.data


class ResizeImageAndBoxesTransform:
    """
    Resizes a numpy image and corresponding bounding boxes to fit the given dimensions.
    """
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img, labels):
        h, w, _ = img.shape
        w_r = self.w / w
        h_r = self.h / h
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        labels[:, 0] *= w_r
        labels[:, 2] *= w_r
        labels[:, 1] *= h_r
        labels[:, 3] *= h_r
        return img, labels


def transform_test_resize(imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), w=640, h=480):
    """
    Function adapted from gluoncv.data.transforms.presets.ssd, resizes the image to a preset size.
    :param imgs:
    :type imgs:
    :param mean:
    :type mean:
    :param std:
    :type std:
    :param w: Desired width of the output tensor.
    :type w: int
    :param h: Desired height of the output tensor.
    :type h: int
    :return:
    :rtype:
    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.imresize(img, w, h)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def transform_test(imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Function dapted from gluoncv.data.transforms.presets.ssd, normalizes and converts image to tensor.
    :param imgs:
    :type imgs:
    :param mean:
    :type mean:
    :param std:
    :type std:
    :return:
    :rtype:
    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs
