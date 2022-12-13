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

import numpy as np
from opendr.engine.target import BoundingBox3DList, TrackingAnnotation3DList
from scipy.optimize import linear_sum_assignment
from opendr.perception.object_tracking_3d.ab3dmot.algorithm.core import convert_3dbox_to_8corner, iou3D
from opendr.perception.object_tracking_3d.ab3dmot.algorithm.kalman_tracker_3d import KalmanTracker3D


class AB3DMOT():
    def __init__(
        self, max_staleness=2, min_updates=3, frame=0,
        state_dimensions=10,  # x, y, z, rotation_y, l, w, h, speed_x, speed_z, angular_speed
        measurement_dimensions=7,  # x, y, z, rotation_y, l, w, h
        state_transition_matrix=None,
        measurement_function_matrix=None,
        covariance_matrix=None,
        process_uncertainty_matrix=None,
        iou_threshold=0.01,
    ):
        super().__init__()

        self.max_staleness = max_staleness
        self.min_updates = min_updates
        self.frame = frame - 1
        self.starting_frame = frame - 1
        self.tracklets = []
        self.last_tracklet_id = 0
        self.iou_threshold = iou_threshold

        self.state_dimensions = state_dimensions
        self.measurement_dimensions = measurement_dimensions
        self.state_transition_matrix = state_transition_matrix
        self.measurement_function_matrix = measurement_function_matrix
        self.covariance_matrix = covariance_matrix
        self.process_uncertainty_matrix = process_uncertainty_matrix

    def update(self, detections: BoundingBox3DList):

        self.frame += 1

        if len(detections) > 0:

            predictions = np.zeros([len(self.tracklets), self.measurement_dimensions])

            for i, tracklet in enumerate(self.tracklets):
                box = tracklet.predict().reshape(-1)[:self.measurement_dimensions]
                predictions[i] = [*box]

            detection_corners = [
                convert_3dbox_to_8corner(np.array([*box.location, box.rotation_y, *box.dimensions]))
                for box in detections.boxes
            ]

            if len(predictions) > 0:
                prediction_corners = [
                    convert_3dbox_to_8corner(p)
                    for p in predictions
                ]
            else:
                prediction_corners = np.zeros((0, 8, 3))

            (
                matched_pairs,
                unmatched_detections,
                unmatched_predictions
            ) = associate(detection_corners, prediction_corners, self.iou_threshold)

            for d, p in matched_pairs:
                self.tracklets[p].update(detections[d], self.frame)

            for d in unmatched_detections:
                self.last_tracklet_id += 1
                tracklet = KalmanTracker3D(
                    detections[d], self.last_tracklet_id, self.frame,
                    self.state_dimensions, self.measurement_dimensions,
                    self.state_transition_matrix, self.measurement_function_matrix,
                    self.covariance_matrix, self.process_uncertainty_matrix
                )
                self.tracklets.append(tracklet)

        old_tracklets = self.tracklets
        self.tracklets = []
        tracked_boxes = []

        for tracklet in old_tracklets:

            if tracklet.staleness(self.frame) < self.max_staleness:
                self.tracklets.append(tracklet)

                if self.frame <= self.min_updates or tracklet.updates >= self.min_updates:
                    tracked_boxes.append(tracklet.tracking_bounding_box_3d(self.frame))

        result = TrackingAnnotation3DList(tracked_boxes)
        return result

    def reset(self):
        self.frame = self.starting_frame
        self.tracklets = []
        self.last_tracklet_id = 0


def associate(detection_corners, prediction_corners, iou_threshold):

    iou_matrix = np.zeros((len(detection_corners), len(prediction_corners)), dtype=np.float32)
    for d, det in enumerate(detection_corners):
        for t, trk in enumerate(prediction_corners):
            iou_matrix[d, t] = iou3D(det, trk)[0]

    detection_match_ids, prediction_match_ids = linear_sum_assignment(-iou_matrix)
    unmatched_detections = []
    unmatched_predictions = []

    for i in range(len(detection_corners)):
        if i not in detection_match_ids:
            unmatched_detections.append(i)

    for i in range(len(prediction_corners)):
        if i not in detection_match_ids:
            unmatched_predictions.append(i)

    matched_pairs = []

    for i in range(len(detection_match_ids)):
        detection_id = detection_match_ids[i]
        prediction_id = prediction_match_ids[i]

        if iou_matrix[detection_id, prediction_id] < iou_threshold:
            unmatched_detections.append(detection_id)
            unmatched_predictions.append(prediction_id)
        else:
            matched_pairs.append([detection_id, prediction_id])

    if len(matched_pairs) <= 0:
        matched_pairs = np.zeros((0, 2), dtype=np.int32)
    else:
        matched_pairs = np.array(matched_pairs, dtype=np.int32)

    return matched_pairs, unmatched_detections, unmatched_predictions
