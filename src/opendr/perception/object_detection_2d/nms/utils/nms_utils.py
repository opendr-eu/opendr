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

import torch
import torchvision
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
import os


def jaccard(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def intersect(box_a, box_b):
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter


def diou(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)
    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    D = (((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((cr - cl) ** 2 + (cb - ct) ** 2 + 1e-7))
    out = inter / area_a if iscrowd else inter / union - D ** 0.9
    return out if use_batch else out.squeeze(0)


def distance(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)

    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    D = (((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((cr - cl) ** 2 + (cb - ct) ** 2 + 1e-7)) ** 0.6
    out = D if iscrowd else D
    return out if use_batch else out.squeeze(0)


def det_matching(scores, dt_boxes, gt_boxes, iou_thres, device='cuda'):
    sorted_indices = torch.argsort(-scores, dim=0)
    labels = torch.zeros(len(dt_boxes))
    if device == 'cuda':
        labels = labels.cuda()
    if gt_boxes.shape[0] == 0:
        return labels.unsqueeze(-1)
    assigned_GT = -torch.ones(len(gt_boxes))
    r = torch.tensor([-1, -1, -1, -1]).float().unsqueeze(0).unsqueeze(0)
    if device == 'cuda':
        r = r.cuda()
    for s in sorted_indices:
        gt_boxes_c = gt_boxes.clone().unsqueeze(0)
        gt_boxes_c[0, assigned_GT > -1, :] = r
        ious = bb_intersection_over_union(boxAs=dt_boxes[s].clone().unsqueeze(0), boxBs=gt_boxes_c)
        annot_iou, annot_box_id = torch.sort(ious.squeeze(), descending=True)
        if annot_box_id.ndim > 0:
            annot_box_id = annot_box_id[0]
            annot_iou = annot_iou[0]
        if annot_iou > iou_thres:
            assigned_GT[annot_box_id] = s
            labels[s] = 1
    return labels.unsqueeze(-1)


def run_coco_eval(dt_file_path=None, gt_file_path=None, only_classes=None, max_dets=None,
                  verbose=False):
    if max_dets is None:
        max_dets = [200, 400, 600, 800, 1000, 1200]
    results = []
    sys.stdout = open(os.devnull, 'w')
    for i in range(len(max_dets)):
        coco = COCO(gt_file_path)
        coco_dt = coco.loadRes(dt_file_path)
        cocoEval = COCOeval(coco, coco_dt, 'bbox')
        cocoEval.params.iouType = 'bbox'
        cocoEval.params.useCats = True
        cocoEval.params.catIds = only_classes
        cocoEval.params.maxDets = [max_dets[i]]
        cocoEval.evaluate()
        results.append([summarize_nms(coco_eval=cocoEval, maxDets=max_dets[i]), max_dets[i]])
        # print(results[i])
    del cocoEval, coco_dt, coco
    sys.stdout = sys.__stdout__
    return results


def summarize_nms(coco_eval=None, maxDets=100):
    def summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        stat_str = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return [mean_s, stat_str]

    def summarizeDets():
        stats = []
        stat, stat_str = summarize(1, maxDets=maxDets)
        stats.append([stat, stat_str])
        stat, stat_str = summarize(1, iouThr=.5, maxDets=maxDets)
        stats.append([stat, stat_str])
        stat, stat_str = summarize(1, iouThr=.75, maxDets=maxDets)
        stats.append([stat, stat_str])
        stat, stat_str = summarize(0, maxDets=maxDets)
        stats.append([stat, stat_str])
        return stats

    coco_eval.accumulate()
    summarized = summarizeDets()
    return summarized


def drop_dets(boxes, scores, keep_ratio=0.85):
    ids = np.arange(len(boxes))
    np.random.shuffle(ids)
    ids_keep = ids[0:int(len(boxes) * keep_ratio)]
    boxes_new = boxes[ids_keep, :]
    scores_new = scores[ids_keep]
    scores_new, scores_new_ids = torch.sort(scores_new, descending=True)
    boxes_new = boxes_new[scores_new_ids]
    return boxes_new, scores_new


def filter_iou_boxes(boxes=None, iou_thres=0.2):
    ious = bb_intersection_over_union(boxes.unsqueeze(1).repeat(1, boxes.shape[0], 1),
                                      boxes.clone().unsqueeze(0).repeat(boxes.shape[0], 1, 1))
    ids_boxes = ious >= iou_thres
    return ids_boxes


def bb_intersection_over_union(boxAs=None, boxBs=None):
    xA = torch.maximum(boxAs[:, :, 0], boxBs[:, :, 0])
    yA = torch.maximum(boxAs[:, :, 1], boxBs[:, :, 1])
    xB = torch.minimum(boxAs[:, :, 2], boxBs[:, :, 2])
    yB = torch.minimum(boxAs[:, :, 3], boxBs[:, :, 3])
    interAreas = torch.maximum(torch.zeros_like(xB), xB - xA + 1) * torch.maximum(torch.zeros_like(yB), yB - yA + 1)
    boxAAreas = (boxAs[:, :, 2] - boxAs[:, :, 0] + 1) * (boxAs[:, :, 3] - boxAs[:, :, 1] + 1)
    boxBAreas = (boxBs[:, :, 2] - boxBs[:, :, 0] + 1) * (boxBs[:, :, 3] - boxBs[:, :, 1] + 1)
    ious = interAreas / (boxAAreas + boxBAreas - interAreas)
    return ious


def compute_class_weights(pos_weights, max_dets=400, dataset_nms=None):
    num_pos = np.ones([len(dataset_nms.classes), 1])
    num_bg = np.ones([len(dataset_nms.classes), 1])
    weights = np.zeros([len(dataset_nms.classes), 2])
    for i in range(len(dataset_nms.src_data)):
        for cls_index in range(len(dataset_nms.classes)):
            num_pos[cls_index] = num_pos[cls_index] + \
                                 min(max_dets, len(dataset_nms.src_data[i]['gt_boxes'][cls_index]))
            num_bg[cls_index] = num_bg[cls_index] + max(0, min(max_dets,
                                                               len(dataset_nms.src_data[i]['dt_boxes'][cls_index])) -
                                                        min(max_dets,
                                                            len(dataset_nms.src_data[i]['gt_boxes'][cls_index])))
    for class_index in range(len(dataset_nms.classes)):
        weights[class_index, 0] = (1 - pos_weights[class_index]) * (num_pos[class_index] +
                                                                    num_bg[class_index]) / num_bg[class_index]
        weights[class_index, 1] = pos_weights[class_index] * (num_pos[class_index] +
                                                              num_bg[class_index]) / num_pos[class_index]
    return weights


def apply_torchNMS(boxes, scores, iou_thres):
    ids_nms = torchvision.ops.nms(boxes, scores, iou_thres)
    scores = scores[ids_nms]
    boxes = boxes[ids_nms]
    return boxes, scores
