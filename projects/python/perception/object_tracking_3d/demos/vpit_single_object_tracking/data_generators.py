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

from opendr.engine.target import TrackingAnnotation3DList
from opendr.perception.object_tracking_3d.datasets.kitti_tracking import LabeledTrackingPointCloudsDatasetIterator
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.run import (
    tracking_boxes_to_lidar
)


def disk_single_object_point_cloud_generator(
    path, track, object_ids, num_point_features=4, cycle=True,
    classes=["Car", "Van", "Truck"],
):
    dataset = LabeledTrackingPointCloudsDatasetIterator(
        path + "/training/velodyne/" + track,
        path + "/training/label_02/" + track + ".txt",
        path + "/training/calib/" + track + ".txt",
    )
    count = len(dataset)

    q = 0

    while q < len(object_ids) or cycle:

        object_id = object_ids[q % len(object_ids)]
        q += 1

        start_frame = -1
        selected_labels = []
        while len(selected_labels) <= 0:
            start_frame += 1

            if start_frame >= len(dataset):
                continue

            point_cloud_with_calibration, labels = dataset[start_frame]
            selected_labels = TrackingAnnotation3DList(
                [label for label in labels if (label.id == object_id)]
            )

        if not selected_labels[0].name in classes:
            continue

        calib = point_cloud_with_calibration.calib
        labels_lidar = tracking_boxes_to_lidar(
            selected_labels, calib, classes=classes
        )
        label_lidar = labels_lidar[0]
        yield point_cloud_with_calibration, label_lidar

        for i in range(start_frame, count):
            point_cloud_with_calibration, labels = dataset[i]
            selected_labels = TrackingAnnotation3DList(
                [label for label in labels if label.id == object_id]
            )

            if len(selected_labels) <= 0:
                break

            yield point_cloud_with_calibration, None


def lidar_point_cloud_generator(lidar):

    while True:
        yield lidar.next()
