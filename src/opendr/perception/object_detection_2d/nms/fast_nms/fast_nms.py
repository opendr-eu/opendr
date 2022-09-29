# Copyright 2020-2022 OpenDR European Project
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

# This file contains code from the CIoU distribution (https://github.com/Zzh-tju/CIoU).
# Copyright (c) 2020 Zheng, Zhaohui.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from opendr.perception.object_detection_2d.nms.utils import NMSCustom
from opendr.perception.object_detection_2d.nms.utils.nms_utils import jaccard
from opendr.engine.target import BoundingBox, BoundingBoxList
import torch
import numpy as np


class FastNMS(NMSCustom):
    def __init__(self, cross_class=False, device='cuda', iou_thres=0.45, top_k=400, post_k=100):
        self.device = device
        self.iou_thres = iou_thres
        self.top_k = top_k
        self.post_k = post_k
        self.cross_class = cross_class

    def set_iou_thres(self, iou_thres=0.45):
        self.iou_thres = iou_thres

    def top_k(self, top_k=400):
        self.top_k = top_k

    def post_k(self, post_k=100):
        self.post_k = post_k

    def set_cross_class(self, cross_class=False):
        self.cross_class = cross_class

    def run_nms(self, boxes=None, scores=None, threshold=0.2, img=None):

        if isinstance(boxes, np.ndarray):
            boxes = torch.tensor(boxes, device=self.device)
        elif torch.is_tensor(boxes):
            if self.device == 'cpu':
                boxes = boxes.cpu()
            elif self.device == 'cuda':
                boxes = boxes.cuda()

        if isinstance(scores, np.ndarray):
            scores = torch.tensor(scores, device=self.device)
        elif torch.is_tensor(scores):
            if self.device == 'cpu':
                scores = scores.cpu()
            elif self.device == 'cuda':
                scores = scores.cuda()

        scores = torch.transpose(scores, dim0=1, dim1=0)
        if self.cross_class:
            [boxes, classes, scores] = cc_fast_nms(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                   top_k=self.top_k, post_k=self.post_k)
        else:
            [boxes, classes, scores] = fast_nms(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                top_k=self.top_k, post_k=self.post_k)

        keep_ids = torch.where(scores > threshold)
        scores = scores[keep_ids].cpu().numpy()
        classes = classes[keep_ids].cpu().numpy()
        boxes = boxes[keep_ids].cpu().numpy()
        bounding_boxes = BoundingBoxList([])
        for idx, box in enumerate(boxes):
            bbox = BoundingBox(left=box[0], top=box[1],
                               width=box[2] - box[0],
                               height=box[3] - box[1],
                               name=classes[idx],
                               score=scores[idx])
            bounding_boxes.data.append(bbox)

        return bounding_boxes, [boxes, classes, scores]


def fast_nms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):
    scores, idx = scores.sort(1, descending=True)
    boxes = boxes[idx, :]

    scores = scores[:, :top_k]
    boxes = boxes[:, :top_k]

    num_classes, num_dets = scores.shape

    boxes = boxes.view(num_classes, num_dets, 4)

    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    keep = (iou_max <= iou_thres)
    keep *= (scores > 0.01)
    classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    scores = scores[keep]

    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]

    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores


def cc_fast_nms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):
    scores, classes = scores.max(dim=0)
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    boxes = boxes[idx]
    scores = scores[idx]
    classes = classes[idx]
    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    maxA, _ = torch.max(iou, dim=0)

    idx_out = torch.where(maxA > iou_thres)
    scores[idx_out] = 0
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores
