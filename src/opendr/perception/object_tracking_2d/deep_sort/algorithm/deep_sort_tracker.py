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
#
# The DeepSortTracker implementation is based on yolov3_deepsort.py, but the detector is removed.
# Detections should be provided by another module.


import numpy as np
from opendr.perception.object_tracking_2d.deep_sort.algorithm.deep_sort import build_tracker
from opendr.engine.data import ImageWithDetections
from opendr.engine.target import TrackingAnnotation, TrackingAnnotationList


class DeepSortTracker(object):
    def __init__(
        self,
        max_dist,
        min_confidence,
        nms_max_overlap,
        max_iou_distance,
        max_age,
        n_init,
        nn_budget,
        device,
    ):

        self.device = device

        self.build_tracker = lambda: build_tracker(
            max_dist,
            min_confidence,
            nms_max_overlap,
            max_iou_distance,
            max_age,
            n_init,
            nn_budget,
            device=device
        )

        self.deepsort = self.build_tracker()
        self.frame = 0

    def infer(self, imageWithDetections: ImageWithDetections, frame_id=None):

        if frame_id is not None:
            self.frame = frame_id

        image = imageWithDetections.numpy().transpose(1, 2, 0)
        detections = imageWithDetections.boundingBoxList

        bbox_xywh = []
        cls_conf = []
        cls_ids = []

        for detection in detections:
            bbox_xywh.append(np.array([
                detection.left,
                detection.top,
                detection.width,
                detection.height,
            ]))
            cls_conf.append(detection.confidence)
            cls_ids.append(detection.name)

        bbox_xywh = np.array(bbox_xywh)
        cls_conf = np.array(cls_conf)
        cls_ids = np.array(cls_ids)

        # bbox dilation just in case bbox too small
        bbox_xywh[:, 3:] *= 1.2

        # do tracking
        outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, image, self.frame <= self.deepsort.tracker.n_init)

        results = []

        # draw boxes for visualization
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, 4]
            confidences = outputs[:, 5]
            classes = outputs[:, 6]

            for bb_xyxy, id, conf, cls_id in zip(bbox_xyxy, identities, confidences, classes):
                bb_tlwh = self.deepsort._xyxy_to_tlwh(bb_xyxy)
                results.append(TrackingAnnotation(
                    cls_id,
                    bb_tlwh[0],
                    bb_tlwh[1],
                    bb_tlwh[2],
                    bb_tlwh[3],
                    id,
                    score=conf,
                    frame=self.frame,
                ))

        self.frame += 1

        return TrackingAnnotationList(results)

    def reset(self):
        self.deepsort = self.build_tracker()
        self.frame = 0
