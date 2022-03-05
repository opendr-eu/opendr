from opendr.perception.object_detection_2d.nms.utils.nms_custom import NMSCustom
from opendr.perception.object_detection_2d.nms.utils.nms_utils import jaccard
from opendr.engine.target import BoundingBox, BoundingBoxList
import torch
import numpy as np


class SoftNMS(NMSCustom):
    def __init__(self, nms_type= 'linear', device='cuda', nms_thres=None, top_k=400, post_k=100):
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

        retained_box = []
        i=0
        while dets.shape[0] > 0:
            max_idx = np.argmax(dets[:, 4], axis=0)
            #dets[[0, max_idx], :] = dets[[max_idx, 0], :]
            scores[i] = dets[0, 4]

            iou = jaccard(dets[:1,:-1], dets[1:,:-1]).triu_(diagonal=0).squeeze(0)

            weight = torch.ones_like(iou)
            if self.nms_type == 'linear':
                weight[iou > self.nms_thres] -= iou[iou > self.nms_thres]
            elif self.nms_type == 'gaussian':
                weight = np.exp(-(iou * iou) / self.nms_thres)

            dets[1:, 4] *= weight
            #retained_idx = torch.where(dets[1:, 4] >= 0)[0]
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





def py_soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):

    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)
