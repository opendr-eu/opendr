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
from filterpy.kalman import KalmanFilter
from opendr.engine.target import BoundingBox3D, TrackingAnnotation3D


class KalmanTracker3D():

    def __init__(
        self,
        boundingBox3D: BoundingBox3D,
        id,
        frame,
        state_dimensions=10,  # x, y, z, rotation_y, l, w, h, speed_x, speed_z, angular_speed
        measurement_dimensions=7,  # x, y, z, rotation_y, l, w, h
        state_transition_matrix=None,
        measurement_function_matrix=None,
        covariance_matrix=None,
        process_uncertainty_matrix=None,
    ):
        super().__init__()

        self.start_frame = frame
        self.last_update_frame = frame
        self.id = id

        self.kalman_filter = KalmanFilter(dim_x=state_dimensions, dim_z=measurement_dimensions)
        self.predictions = []
        self.updates = 0

        if state_transition_matrix is None:
            state_transition_matrix = np.eye(state_dimensions, dtype=np.float32)
            state_transition_matrix[0, -3] = 1
            state_transition_matrix[1, -2] = 1
            state_transition_matrix[2, -1] = 1

        if measurement_function_matrix is None:
            measurement_function_matrix = np.eye(
                measurement_dimensions, state_dimensions, dtype=np.float32
            )

        if covariance_matrix is None:
            covariance_matrix = np.eye(
                state_dimensions, state_dimensions, dtype=np.float32
            ) * 10
            covariance_matrix[7:, 7:] *= 1000

        if process_uncertainty_matrix is None:
            process_uncertainty_matrix = np.eye(
                state_dimensions, state_dimensions, dtype=np.float32
            )
            process_uncertainty_matrix[7:, 7:] *= 0.01

        self.kalman_filter.F = state_transition_matrix
        self.kalman_filter.H = measurement_function_matrix
        self.kalman_filter.P = covariance_matrix
        self.kalman_filter.Q = process_uncertainty_matrix

        location = boundingBox3D.data["location"]
        dimensions = boundingBox3D.data["dimensions"]
        rotation_y = boundingBox3D.data["rotation_y"]

        # [x, y, z, rotation_y, l, w, h]
        self.kalman_filter.x[:measurement_dimensions] = np.array([
            *location, rotation_y, *dimensions
        ]).reshape(-1, 1)

        self.name = boundingBox3D.name
        self.bbox2d = boundingBox3D.bbox2d
        self.action = boundingBox3D.action
        self.alpha = boundingBox3D.alpha
        self.truncated = boundingBox3D.truncated
        self.occluded = boundingBox3D.occluded
        self.confidence = boundingBox3D.confidence

    def update(self, boundingBox3D: BoundingBox3D, frame):

        self.last_update_frame = frame
        self.updates += 1

        location = boundingBox3D.data["location"]
        dimensions = boundingBox3D.data["dimensions"]
        rotation_y = boundingBox3D.data["rotation_y"]

        self.name = boundingBox3D.name
        self.bbox2d = boundingBox3D.bbox2d
        self.action = boundingBox3D.action
        self.alpha = boundingBox3D.alpha
        self.truncated = boundingBox3D.truncated
        self.occluded = boundingBox3D.occluded
        self.confidence = boundingBox3D.confidence

        rotation_y = normalize_angle(rotation_y)
        predicted_rotation_y = self.kalman_filter.x[3]

        if (
            abs(rotation_y - predicted_rotation_y) >= np.pi / 2 and
            abs(rotation_y - predicted_rotation_y) <= np.pi * 1.5
        ):
            predicted_rotation_y = normalize_angle(predicted_rotation_y + np.pi)

        if abs(rotation_y - predicted_rotation_y) >= np.pi * 1.5:
            if rotation_y > 0:
                predicted_rotation_y += np.pi * 2
            else:
                predicted_rotation_y -= np.pi * 2

        self.kalman_filter.x[3] = predicted_rotation_y

        self.kalman_filter.update(np.array([
            *location, rotation_y, *dimensions
        ]))

    def predict(self) -> np.ndarray:
        self.kalman_filter.predict()
        self.kalman_filter.x[3] = normalize_angle(self.kalman_filter.x[3])
        self.predictions.append(self.kalman_filter.x)

        return self.kalman_filter.x

    def tracking_bounding_box_3d(self, frame):
        return TrackingAnnotation3D(
            self.name, self.truncated, self.occluded,
            self.alpha, self.bbox2d,
            self.kalman_filter.x[4:].reshape(-1),
            self.kalman_filter.x[:3].reshape(-1),
            float(self.kalman_filter.x[3]),
            self.id,
            self.confidence,
            frame,
        )

    def age(self, frame):
        return frame - self.start_frame

    def staleness(self, frame):
        return frame - self.last_update_frame


def normalize_angle(angle):
    if angle >= np.pi:
        angle -= np.pi * 2
    if angle < -np.pi:
        angle += np.pi * 2
    return angle
