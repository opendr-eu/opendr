import numpy as np

from .processing.bbox_transform import bbox_overlaps

from opendr.perception.object_detection_2d.utils.eval_utils import CustomEvalMetric


class FaceDetectionRecallMetric(CustomEvalMetric):
    """
    Recall metric, adapted from RetinaFace code.
    """
    def __init__(self, classes=None):
        super().__init__(classes)
        self.overall = [0.0, 0.0]

    def update(self, det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult=None):
        batch_det_boxes, batch_det_labels, batch_det_scores, batch_gt_boxes, batch_gt_labels, batch_gt_difficult = \
            self._input_check(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult)

        n_batch = det_boxes.shape[0]
        for n in range(n_batch):
            det_boxes = batch_det_boxes[n, :, :]
            gt_boxes = batch_gt_boxes[n, :, :]
            overlaps = bbox_overlaps(det_boxes.astype(np.float), gt_boxes.astype(np.float))
            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            if det_boxes.shape[0] > 0:
                _gt_overlaps = overlaps.max(axis=0)
                for j in range(len(_gt_overlaps)):
                    if _gt_overlaps[j] > 0.5:
                        continue

                # append recorded IoU coverage level
                found = (_gt_overlaps > 0.5).sum()
                # recall = found / float(gt_boxes.shape[0])

                self.overall[0] += found
                self.overall[1] += gt_boxes.shape[0]
                # recall_all = float(self.overall[0]) / max(1, self.overall[1])

    def get(self):
        return ["recall"], [float(self.overall[0]) / max(1, self.overall[1])]
