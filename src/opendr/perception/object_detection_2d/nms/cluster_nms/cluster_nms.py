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
from opendr.perception.object_detection_2d.nms.utils.nms_utils import jaccard, diou, distance
from opendr.engine.target import BoundingBox, BoundingBoxList
import numpy as np
import torch


class ClusterNMS(NMSCustom):
    def __init__(self, nms_type='default', cross_class=True, device='cuda', iou_thres=0.45, top_k=400, post_k=100):
        self.device = device
        self.nms_types = ['default', 'diou', 'spm', 'spm_dist', 'spm_dist_weighted']
        if nms_type not in self.nms_types:
            raise ValueError('Type: ' + nms_type + ' of Cluster-NMS is not supported.')
        else:
            self.nms_type = nms_type
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

    def set_type(self, nms_type=None):
        if nms_type not in self.nms_types:
            raise ValueError('Type: ' + nms_type + ' of Cluster-NMS is not supported.')
        else:
            self.nms_type = nms_type

    def set_cross_class(self, cross_class=True):
        self.cross_class = cross_class

    def run_nms(self, boxes=None, scores=None, img=None, threshold=0.2):

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

        if self.nms_type == 'default':
            if self.cross_class:
                [boxes, classes, scores] = cc_cluster_nms_default(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                                  top_k=self.top_k, post_k=self.post_k)
            else:
                [boxes, classes, scores] = cluster_nms_default(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                               top_k=self.top_k, post_k=self.post_k)
        elif self.nms_type == 'diou':
            if self.cross_class:
                [boxes, classes, scores] = cc_cluster_diounms(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                              top_k=self.top_k, post_k=self.post_k)
            else:
                [boxes, classes, scores] = cluster_diounms(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                           top_k=self.top_k, post_k=self.post_k)
        elif self.nms_type == 'spm':
            if self.cross_class:
                [boxes, classes, scores] = cc_cluster_SPM_nms(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                              top_k=self.top_k, post_k=self.post_k)
            else:
                [boxes, classes, scores] = cluster_SPM_nms(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                           top_k=self.top_k, post_k=self.post_k)
        elif self.nms_type == 'spm_dist':
            if self.cross_class:
                [boxes, classes, scores] = cc_cluster_SPM_dist_nms(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                                   top_k=self.top_k, post_k=self.post_k)
            else:
                [boxes, classes, scores] = cluster_SPM_dist_nms(boxes=boxes, scores=scores, iou_thres=self.iou_thres,
                                                                top_k=self.top_k, post_k=self.post_k)

        elif self.nms_type == 'spm_dist_weighted':
            if self.cross_class:
                [boxes, classes, scores] = cc_cluster_SPM_dist_weighted_nms(boxes=boxes, scores=scores,
                                                                            iou_thres=self.iou_thres,
                                                                            top_k=self.top_k,
                                                                            post_k=self.post_k)
            else:
                [boxes, classes, scores] = cluster_SPM_dist_weighted_nms(boxes=boxes, scores=scores,
                                                                         iou_thres=self.iou_thres,
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


def cc_cluster_nms_default(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):
    # Collapse all the classes into 1

    scores, classes = scores.max(dim=0)
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    boxes = boxes[idx]
    scores = scores[idx]
    classes = classes[idx]
    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_thres).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break

    idx_out = torch.where(maxA > iou_thres)
    scores[idx_out] = 0
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores


def cluster_nms_default(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, idx = scores.sort(1, descending=True)
    idx = idx[:top_k]
    scores = scores[:top_k]
    boxes = boxes[idx, :]

    num_classes, num_dets = scores.shape
    boxes = boxes.view(num_classes, num_dets, 4)
    _, classes = scores.max(dim=0)
    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    B = iou
    maxA = None
    for i in range(200):
        A = B
        maxA, _ = A.max(dim=1)
        E = (maxA <= iou_thres).float().unsqueeze(2).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break
    keep = (scores > 0.00)
    discard = (maxA > iou_thres)
    scores[discard] = 0
    # Assign each kept detection to its corresponding class
    boxes = boxes[keep]
    scores = scores[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores


def cc_cluster_diounms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, classes = scores.max(dim=0)
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    boxes = boxes[idx]
    scores = scores[idx]
    classes = classes[idx]
    iou = diou(boxes, boxes).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_thres).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break

    idx_out = torch.where(maxA > iou_thres)
    scores[idx_out] = 0
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores


def cluster_diounms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, idx = scores.sort(1, descending=True)
    idx = idx[:top_k]
    scores = scores[:top_k]
    boxes = boxes[idx, :]

    num_classes, num_dets = scores.shape
    boxes = boxes.view(num_classes, num_dets, 4)
    _, classes = scores.max(dim=0)

    iou = diou(boxes, boxes).triu_(diagonal=1)
    B = iou
    maxA = None
    for i in range(200):
        A = B
        maxA, _ = A.max(dim=1)
        E = (maxA <= iou_thres).float().unsqueeze(2).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break
    keep = (scores > 0.00)
    discard = (maxA > iou_thres)
    scores[discard] = 0
    # Assign each kept detection to its corresponding class
    boxes = boxes[keep]
    scores = scores[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]

    return boxes, classes, scores


def cc_cluster_SPM_nms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, classes = scores.max(dim=0)
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    boxes = boxes[idx]
    scores = scores[idx]
    classes = classes[idx]
    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_thres).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break
    scores = torch.prod(torch.exp(-B ** 2 / 0.2), 0) * scores

    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores


def cluster_SPM_nms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, idx = scores.sort(1, descending=True)
    idx = idx[:top_k]
    scores = scores[:top_k]
    boxes = boxes[idx, :]

    num_classes, num_dets = scores.shape
    boxes = boxes.view(num_classes, num_dets, 4)
    _, classes = scores.max(dim=0)

    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = A.max(dim=1)
        E = (maxA <= iou_thres).float().unsqueeze(2).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break
    keep = (scores > 0.00)
    scores = torch.prod(torch.exp(-B ** 2 / 0.2), 1) * scores
    # Assign each kept detection to its corresponding class
    boxes = boxes[keep]
    scores = scores[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores


def cc_cluster_SPM_dist_nms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, classes = scores.max(dim=0)
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    boxes = boxes[idx]
    scores = scores[idx]
    classes = classes[idx]
    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_thres).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break
    D = distance(boxes, boxes)
    X = (B >= 0).float()
    scores = torch.prod(torch.min(torch.exp(-B ** 2 / 0.2) + D * ((B > 0).float()), X), 0) * scores

    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores


def cluster_SPM_dist_nms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, idx = scores.sort(1, descending=True)
    idx = idx[:top_k]
    scores = scores[:top_k]
    boxes = boxes[idx, :]

    num_classes, num_dets = scores.shape
    boxes = boxes.view(num_classes, num_dets, 4)
    _, classes = scores.max(dim=0)

    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = A.max(dim=1)
        E = (maxA <= iou_thres).float().unsqueeze(2).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break
    D = distance(boxes, boxes)
    X = (B >= 0).float()
    keep = (scores > 0.00)
    scores = torch.prod(torch.min(torch.exp(-B ** 2 / 0.2) + D * ((B > 0).float()), X), 1) * scores

    # Assign each kept detection to its corresponding class
    boxes = boxes[keep]
    scores = scores[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]

    return boxes, classes, scores


def cc_cluster_SPM_dist_weighted_nms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, classes = scores.max(dim=0)
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    boxes = boxes[idx]
    scores = scores[idx]
    classes = classes[idx]
    n = len(scores)
    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_thres).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break
    D = distance(boxes, boxes)
    X = (B >= 0).float()
    scores = torch.prod(torch.min(torch.exp(-B ** 2 / 0.2) + D * ((B > 0).float()), X), 0) * scores
    eye = torch.eye(n)
    if boxes.device.type == 'cuda':
        eye = eye.cuda()
    weights = (B * (B > 0.8).float() + eye) * (scores.reshape((1, n)))
    xx1 = boxes[:, 0].expand(n, n)
    yy1 = boxes[:, 1].expand(n, n)
    xx2 = boxes[:, 2].expand(n, n)
    yy2 = boxes[:, 3].expand(n, n)

    weightsum = weights.sum(dim=1)
    xx1 = (xx1 * weights).sum(dim=1) / (weightsum)
    yy1 = (yy1 * weights).sum(dim=1) / (weightsum)
    xx2 = (xx2 * weights).sum(dim=1) / (weightsum)
    yy2 = (yy2 * weights).sum(dim=1) / (weightsum)
    boxes = torch.stack([xx1, yy1, xx2, yy2], 1)

    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]
    return boxes, classes, scores


def cluster_SPM_dist_weighted_nms(boxes=None, scores=None, iou_thres=0.45, top_k=400, post_k=200):

    scores, idx = scores.sort(1, descending=True)
    idx = idx[:top_k]
    scores = scores[:top_k]
    boxes = boxes[idx, :]

    num_classes, num_dets = scores.shape
    boxes = boxes.view(num_classes, num_dets, 4)
    _, classes = scores.max(dim=0)

    iou = jaccard(boxes, boxes).triu_(diagonal=1)
    B = iou
    A = None
    for i in range(200):
        A = B
        maxA, _ = A.max(dim=1)
        E = (maxA <= iou_thres).float().unsqueeze(2).expand_as(A)
        B = iou.mul(E)
        if A.equal(B):
            break
    D = distance(boxes, boxes)
    X = (B >= 0).float()
    keep = (scores > 0.0)

    scores = torch.prod(torch.min(torch.exp(-B ** 2 / 0.2) + D * ((B > 0).float()), X), 1) * scores

    E = keep.float().unsqueeze(2).expand_as(A)
    B = iou.mul(E)
    _, n = scores.size()
    eye = torch.eye(n).expand(num_classes, n, n)
    if boxes.device.type == 'cuda':
        eye = eye.cuda()
    weights = (B * (B > 0.8).float() + eye) * (
        scores.unsqueeze(2).expand(num_classes, n, n))
    xx1 = boxes[:, :, 0].unsqueeze(1).expand(num_classes, n, n)
    yy1 = boxes[:, :, 1].unsqueeze(1).expand(num_classes, n, n)
    xx2 = boxes[:, :, 2].unsqueeze(1).expand(num_classes, n, n)
    yy2 = boxes[:, :, 3].unsqueeze(1).expand(num_classes, n, n)

    weightsum = weights.sum(dim=2)
    xx1 = (xx1 * weights).sum(dim=2) / (weightsum)
    yy1 = (yy1 * weights).sum(dim=2) / (weightsum)
    xx2 = (xx2 * weights).sum(dim=2) / (weightsum)
    yy2 = (yy2 * weights).sum(dim=2) / (weightsum)
    boxes = torch.stack([xx1, yy1, xx2, yy2], 2)

    # Assign each kept detection to its corresponding class
    classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]
    boxes = boxes[keep]
    scores = scores[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:post_k]
    scores = scores[:post_k]
    classes = classes[idx]
    boxes = boxes[idx]

    return boxes, classes, scores
