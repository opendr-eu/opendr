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

# MIT License
#
# Copyright (c) 2020 DocF
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from opendr.perception.object_detection_2d.nms.utils import NMSCustom
from opendr.perception.object_detection_2d.nms.utils.nms_utils import jaccard
from opendr.engine.target import BoundingBox, BoundingBoxList
import torch
import numpy as np


class SoftNMS(NMSCustom):
    def __init__(self, nms_type='linear', device='cuda', nms_thres=None, top_k=400, post_k=100):
        self.nms_types = ['linear', 'gaussian']
        if nms_type not in self.nms_types:
            raise ValueError('Type: ' + nms_type + ' of Soft-NMS is not supported.')
        else:
            self.nms_type = nms_type
        if nms_thres is None:
            if nms_type == 'linear':
                nms_thres = 0.3
            elif nms_type == 'gaussian':
                nms_thres = 0.5
        self.device = device
        self.nms_thres = nms_thres
        self.top_k = top_k
        self.post_k = post_k

    def nms_thres(self, nms_thres=0.45):
        self.nms_thres = nms_thres

    def set_top_k(self, top_k=400):
        self.top_k = top_k

    def set_post_k(self, post_k=100):
        self.post_k = post_k

    def set_nms_type(self, nms_type='linear'):
        if nms_type not in self.nms_types:
            raise ValueError('Type: ' + nms_type + ' of Soft-NMS is not supported.')
        else:
            self.nms_type = nms_type

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

        scores, classes = scores.max(dim=1)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:self.top_k]
        boxes = boxes[idx]
        scores = scores[idx]
        classes = classes[idx]

        dets = torch.cat((boxes, scores.unsqueeze(-1)), dim=1)

        i = 0
        while dets.shape[0] > 0:
            scores[i] = dets[0, 4]
            iou = jaccard(dets[:1, :-1], dets[1:, :-1]).triu_(diagonal=0).squeeze(0)
            weight = torch.ones_like(iou)
            if self.nms_type == 'linear':
                weight[iou > self.nms_thres] -= iou[iou > self.nms_thres]
            elif self.nms_type == 'gaussian':
                weight = np.exp(-(iou * iou) / self.nms_thres)

            dets[1:, 4] *= weight
            dets = dets[1:, :]
            i = i + 1
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
