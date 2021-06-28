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

import numpy as np
import mxnet as mx

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def as_numpy(a):
    """Convert a (list of) mx.NDArray into numpy.ndarray"""
    if isinstance(a, (list, tuple)):
        out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
        return np.concatenate(out, axis=0)
    elif isinstance(a, mx.nd.NDArray):
        a = a.asnumpy()
    return a


def find_intersection(set1, set2):
    lower_bounds = np.maximum(np.expand_dims(set1[:, :2], 1), np.expand_dims(set2[:, :2], 0))
    upper_bounds = np.minimum(np.expand_dims(set1[:, 2:], 1), np.expand_dims(set2[:, 2:], 0))
    intersection_dims = np.clip(upper_bounds - lower_bounds, a_min=0, a_max=None)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set1, set2):
    intersection = find_intersection(set1, set2)

    areas_set1 = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1])
    areas_set2 = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1])

    union = np.expand_dims(areas_set1, 1) + np.expand_dims(areas_set2, 0) - intersection
    return intersection / union


class DetectionEvalMetric:
    def __init__(self, classes, n_val_images=None):
        self.classes = classes
        self.n_val_images = n_val_images
        self.results = {'map': 0}

    def reset(self):
        raise NotImplementedError("DetectionEvalMetric objects must implement reset() function")

    def update(self, det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult=None):
        raise NotImplementedError("DetectionEvalMetric objects must implement update() function")

    def get(self):
        return list(self.results.keys()), list(self.results.values())

    def _input_check(self, det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult):
        return as_numpy(det_boxes), as_numpy(det_labels), as_numpy(det_scores), \
               as_numpy(gt_boxes), as_numpy(gt_labels), as_numpy(gt_difficult)


class MeanAveragePrecision(DetectionEvalMetric):
    def __init__(self, classes, n_val_images):
        super().__init__(classes)
        self.det_boxes = []
        self.det_labels = []
        self.det_scores = []
        self.gt_boxes = []
        self.gt_labels = []
        self.gt_difficult = []
        self.n_val_images = n_val_images
        self.n_classes = len(classes)
        self.average_precisions = np.zeros((self.n_classes))
        self.results = {"map": 0}

    def reset(self):
        self.det_boxes = []
        self.det_labels = []
        self.det_scores = []
        self.gt_boxes = []
        self.gt_labels = []
        self.gt_difficult = []
        self.average_precisions = np.zeros((self.n_classes))

    def update(self, det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult=None):
        det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult = \
            self._input_check(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult)

        self.det_boxes.extend(det_boxes)
        self.det_labels.extend(det_labels)
        self.det_scores.extend(det_scores)
        self.gt_boxes.extend(gt_boxes)
        self.gt_labels.extend(gt_labels)
        if gt_difficult is not None:
            self.gt_difficult.extend(np.asarray(gt_difficult))
        else:
            self.gt_difficult.extend(np.zeros((len(gt_labels),)))

    def get(self):
        result = {"map": 0}

        assert len(self.det_boxes) == len(self.gt_boxes) == len(self.det_labels) == len(self.gt_labels) == len(
            self.det_scores) == len(self.gt_difficult) == self.n_val_images, "Unequal number of detected and true objects"

        gt_images = []
        for i in range(len(self.gt_labels)):
            gt_images.extend([i] * len(self.gt_labels[i]))
        gt_images = np.asarray(gt_images)
        gt_boxes = np.concatenate(self.gt_boxes, axis=0)
        gt_labels = np.concatenate(self.gt_labels, axis=0)
        if gt_labels.ndim > 1:
            gt_labels = gt_labels[:, 0]
        gt_difficult = np.concatenate(self.gt_difficult, axis=0)
        if gt_difficult.ndim > 1:
            gt_difficult = gt_difficult[:, 0]

        det_images = []
        for i in range(len(self.det_labels)):
            det_images.extend([i] * len(self.det_labels[i]))
        det_images = np.asarray(det_images)
        det_boxes = np.concatenate(self.det_boxes, axis=0)
        det_labels = np.concatenate(self.det_labels, axis=0)
        if det_labels.ndim > 1:
            det_labels = det_labels[:, 0]
        det_scores = np.concatenate(self.det_scores, axis=0)
        if det_scores.ndim > 1:
            det_scores = det_scores[:, 0]

        average_precisions = np.zeros((self.n_classes), dtype=np.float)
        for c in range(self.n_classes):
            gt_c_labels = np.where(gt_labels == c)[0]
            gt_class_images = gt_images[gt_c_labels]
            gt_class_boxes = gt_boxes[gt_c_labels]
            gt_class_difficult = gt_difficult[gt_c_labels]
            n_easy_class_objects = np.sum(1 - gt_class_difficult, dtype=np.uint32)

            gt_class_boxes_detected = np.zeros((gt_class_difficult.shape[0]), dtype=np.uint8)

            det_c_labels = np.where(det_labels == c)[0]
            det_class_images = det_images[det_c_labels]
            det_class_boxes = det_boxes[det_c_labels]
            det_class_scores = det_scores[det_c_labels]
            n_class_detections = det_class_boxes.shape[0]
            if n_class_detections == 0:
                continue

            # sort detection in decreasing order of confidence
            sort_ind = np.argsort(-det_class_scores, axis=0)
            # det_class_scores = det_class_scores[sort_ind]
            det_class_boxes = det_class_boxes[sort_ind]
            det_class_images = det_class_images[sort_ind]

            true_positives = np.zeros((n_class_detections), dtype=np.float)
            false_positives = np.zeros((n_class_detections), dtype=np.float)
            for d in range(n_class_detections):
                this_detection_box = np.expand_dims(det_class_boxes[d], 0)
                this_image = det_class_images[d]

                object_boxes = gt_class_boxes[gt_class_images == this_image]
                object_difficulties = gt_class_difficult[gt_class_images == this_image]
                if object_boxes.shape[0] == 0:
                    false_positives[d] = 1
                    continue

                overlaps = find_jaccard_overlap(this_detection_box, object_boxes)[0, :]
                ind = np.argmax(overlaps, axis=0)
                max_overlap = overlaps[ind]
                original_ind = np.arange(0, gt_class_boxes.shape[0])[gt_class_images == this_image][ind]

                if max_overlap > 0.5:
                    if object_difficulties[ind] == 0:
                        if gt_class_boxes_detected[original_ind] == 0:
                            true_positives[d] = 1
                            gt_class_boxes_detected[original_ind] = 1
                        else:
                            false_positives[d] = 1
                else:
                    false_positives[d] = 1
            cumul_true_positives = np.cumsum(true_positives, axis=0)
            cumul_false_positives = np.cumsum(false_positives, axis=0)
            cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
            cumul_recall = cumul_true_positives / n_easy_class_objects

            recall_thresholds = np.arange(0.5, 1., step=.05)
            precisions = np.zeros((len(recall_thresholds)), dtype=np.float)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t = cumul_recall >= t
                if np.any(recalls_above_t):
                    precisions[i] = cumul_precision[recalls_above_t].max()
                else:
                    precisions[i] = 0
            average_precisions[c] = np.mean(precisions)

        mean_average_precision = np.mean(average_precisions)
        result["map"] = mean_average_precision
        self.average_precisions = average_precisions
        self.results = result
        # for c, class_name in enumerate(self.classes):
        #     result["ap_{}".format(class_name)] = average_precisions[c]
        return list(result.keys()), list(result.values())


class DetectionDatasetCOCOEval(DetectionEvalMetric):
    def __init__(self, classes, data_shape, score_threshold=0.1):
        super().__init__(classes)
        if isinstance(data_shape, tuple):
            self.width = data_shape[0]
            self.height = data_shape[1]
        else:
            self.width = data_shape
            self.height = data_shape
        self.n_classes = len(classes)
        self.score_threshold = score_threshold

        self.ann_dict_gt = {}
        self.annotations_gt = []
        self.ann_id = 0

        self.detections = []

        self.images = []
        self.img_id = 0
        self.results = {}

    def reset(self):
        self.ann_dict_gt = {}
        self.annotations_gt = []
        self.ann_id = 0

        self.detections = []

        self.images = []
        self.img_id = 0
        self.results = {}

    def update(self, det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult=None):
        det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult = \
            self._input_check(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult)
        n_batch = det_boxes.shape[0]

        for idx in range(n_batch):
            image = {}
            image['id'] = self.img_id
            self.img_id += 1
            image['height'] = str(self.width)
            image['width'] = str(self.height)
            image['filename'] = ''
            self.images.append(image)

            img_dets = []
            for box_idx, box in enumerate(det_boxes[idx, :, :]):
                cls = det_labels[idx, box_idx]
                if not np.isscalar(cls):
                    cls = cls[0]
                if cls < 0:
                    continue
                score = det_scores[idx, box_idx]
                if not np.isscalar(score):
                    score = score[0]
                if score < self.score_threshold:
                    continue
                img_dets.append(np.asarray([image['id'], box[0], box[1], box[2] - box[0], box[3] - box[1], score, cls]))
            self.detections.append(np.asarray(img_dets))

            for box_idx, box in enumerate(gt_boxes[idx, :, :]):
                cls = gt_labels[idx, box_idx]
                if not np.isscalar(cls):
                    cls = cls[0]
                if cls < 0:
                    continue

                ann = {}
                ann['id'] = self.ann_id
                self.ann_id += 1
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = int(cls)
                ann['iscrowd'] = 0
                ann['area'] = int((box[2] - box[0]) * (box[3] - box[1]))
                ann['bbox'] = [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]
                self.annotations_gt.append(ann)

    def get(self):
        self.ann_dict_gt['images'] = self.images
        self.ann_dict_gt['categories'] = [{'id': idx, 'name': name} for idx, name in enumerate(self.classes)]
        self.ann_dict_gt['annotations'] = self.annotations_gt

        detections_array = np.concatenate(self.detections, axis=0)

        _COCO = COCO()
        _COCO.dataset = self.ann_dict_gt
        _COCO.createIndex()
        _COCO.img_ann_map = _COCO.imgToAnns
        _COCO.cat_img_map = _COCO.catToImgs
        coco_dt = _COCO.loadRes(detections_array)
        coco_eval = COCOeval(_COCO, coco_dt)
        coco_eval.params.useSegm = False
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        return list(self.results.keys()), list(self.results.values())

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
              '~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        print('{}: {:.1f}'.format('mAP', 100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind, 0, 2]
            ap = np.mean(precision[precision > -1])
            self.results[cls] = ap
            print('{}: {:.1f}'.format(cls, 100 * ap))
        self.results["map"] = np.mean(ap_default)

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()


class CustomEvalMetric(DetectionEvalMetric):
    """ Dummy DetectionEvalMetric subclass for datasets which implement their own evaluation scheme.
     Detections and corresponding groundtruths are made available via the get_dts() and get_gts() functions. """
    def __init__(self, classes):
        super().__init__(classes)
        self.det_boxes = []
        self.det_labels = []
        self.det_scores = []
        self.gt_boxes = []
        self.gt_labels = []
        self.gt_difficult = []
        if classes is None:
            self.n_classes = 1
        else:
            self.n_classes = len(classes)
        self.results = {"map": 0}

    def reset(self):
        self.det_boxes = []
        self.det_labels = []
        self.det_scores = []
        self.gt_boxes = []
        self.gt_labels = []
        self.gt_difficult = []
        self.results = {"map": 0}

    def update(self, det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult=None):
        det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult = \
            self._input_check(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult)

        self.det_boxes.extend(det_boxes)
        self.det_labels.extend(det_labels)
        self.det_scores.extend(det_scores)
        self.gt_boxes.extend(gt_boxes)
        self.gt_labels.extend(gt_labels)
        if gt_difficult is not None:
            self.gt_difficult.extend(np.asarray(gt_difficult))
        else:
            self.gt_difficult.extend(np.zeros((len(gt_labels),)))

    def get(self):
        print('This evaluation class does not compute any metrics but is intended for use with custom evaluation schemes,'
              'by accumulating the detections and corresponding groundtruth boxes.')
        return list(self.results.keys()), list(self.results.values())

    def get_dts(self):
        """

        :return: det_boxes, det_labels, det_scores
        :rtype: list(n_images, n_boxes, 4), list(n_images, n_boxes), list(n_images, n_boxes)
        """
        return self.det_boxes, self.det_labels, self.det_boxes

    def get_gts(self):
        """

        :return: gt_boxes, gt_labels, gt_difficult
        :rtype: list(n_images, n_boxes, 4), list(n_images, n_boxes), list(n_images, n_boxes)
        """
        return self.gt_boxes, self.gt_labels, self.gt_difficult
