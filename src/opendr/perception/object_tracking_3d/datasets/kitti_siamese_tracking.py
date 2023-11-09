# Copyright 2020-2023 OpenDR European Project
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

# KITTI tracking data format:
# (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry,[score])
# KITTI detection data format (with frame):
# (frame,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry,[score])

import os
from opendr.engine.target import (
    BoundingBox3D,
    BoundingBox3DList,
    TrackingAnnotation3D,
    TrackingAnnotation3DList,
)
from opendr.engine.data import PointCloudWithCalibration
from opendr.engine.datasets import DatasetIterator
import numpy as np

from opendr.perception.object_detection_3d.datasets.kitti import parse_calib


class SiameseTrackingDatasetIterator(DatasetIterator):
    def __init__(
        self,
        lidar_paths,
        label_paths,
        calib_paths,
        labels_format="tracking",  # detection, tracking
        classes=["Car"],  # detection, tracking
        num_point_features=4,
        samples_per_object=200,
    ):
        super().__init__()

        self.lidar_paths = lidar_paths
        self.label_paths = label_paths
        self.calib_paths = calib_paths
        self.num_point_features = num_point_features

        self.lidar_files = [
            sorted(os.listdir(lidar_path)) for lidar_path in self.lidar_paths
        ]
        labels_and_max_ids = [
            load_tracking_file(label_path, "tracking", labels_format,)
            for label_path in self.label_paths
        ]

        self.labels = [x[0] for x in labels_and_max_ids]
        self.max_ids = [x[1] for x in labels_and_max_ids]

        self.calibs = [parse_calib(calib_path) for calib_path in self.calib_paths]

        self.track_target_search_frames_with_id = []

        for track_id, (track_labels, max_id) in enumerate(labels_and_max_ids):
            for object_id in range(max_id + 1):
                frames_with_current_object = []
                for frame, frame_labels in enumerate(track_labels):
                    for label in frame_labels:
                        if (
                            (label.id == object_id) and
                            (label.id >= 0) and
                            label.name in classes
                        ):
                            frames_with_current_object.append(frame)

                object_samples = []

                for a in frames_with_current_object:
                    for b in frames_with_current_object:
                        object_samples.append(
                            (track_id, a, b, object_id)
                        )

                np.random.shuffle(object_samples)

                if len(object_samples) > samples_per_object:
                    object_samples = object_samples[:samples_per_object]

                if len(object_samples) > 0:
                    self.track_target_search_frames_with_id.extend(
                        object_samples
                    )

        print()

    def __getitem__(self, idx):

        (
            track_id,
            target_frame_id,
            search_frame_id,
            object_id,
        ) = self.track_target_search_frames_with_id[idx]

        target_points = np.fromfile(
            os.path.join(
                self.lidar_paths[track_id], self.lidar_files[track_id][target_frame_id]
            ),
            dtype=np.float32,
            count=-1,
        ).reshape([-1, self.num_point_features])

        search_points = np.fromfile(
            os.path.join(
                self.lidar_paths[track_id], self.lidar_files[track_id][search_frame_id]
            ),
            dtype=np.float32,
            count=-1,
        ).reshape([-1, self.num_point_features])

        target_label = None

        for x in self.labels[track_id][target_frame_id]:
            if x.id == object_id:
                target_label = x
                break
        # target_label = [
        #     x for x in self.labels[track_id][target_frame_id] if x.id == object_id
        # ][0]

        search_label = None
        for x in self.labels[track_id][search_frame_id]:
            if x.id == object_id:
                search_label = x
                break

        # search_label = [
        #     x for x in self.labels[track_id][search_frame_id] if x.id == object_id
        # ][0]

        result = (
            PointCloudWithCalibration(target_points, self.calibs[track_id], None),
            PointCloudWithCalibration(search_points, self.calibs[track_id], None),
            target_label,
            search_label,
        )

        return result

    def __len__(self):
        return len(self.track_target_search_frames_with_id)


class SiameseTripletTrackingDatasetIterator(DatasetIterator):
    def __init__(
        self,
        lidar_paths,
        label_paths,
        calib_paths,
        labels_format="tracking",  # detection, tracking
        classes=["Car"],  # detection, tracking
        num_point_features=4,
    ):
        super().__init__()

        self.lidar_paths = lidar_paths
        self.label_paths = label_paths
        self.calib_paths = calib_paths
        self.num_point_features = num_point_features

        self.lidar_files = [
            sorted(os.listdir(lidar_path)) for lidar_path in self.lidar_paths
        ]
        labels_and_max_ids = [
            load_tracking_file(label_path, "tracking", labels_format,)
            for label_path in self.label_paths
        ]

        self.labels = [x[0] for x in labels_and_max_ids]
        self.max_ids = [x[1] for x in labels_and_max_ids]

        self.calibs = [parse_calib(calib_path) for calib_path in self.calib_paths]

        self.track_target_search_frames_with_id_other_frame_id = []

        for track_id, (track_labels, max_id) in enumerate(labels_and_max_ids):

            for object_id in range(max_id + 1):
                frames_with_current_object = []
                other_objects = []
                for frame, frame_labels in enumerate(track_labels):
                    for label in frame_labels:
                        if (label.id >= 0) and label.name in classes:
                            if label.id == object_id:
                                frames_with_current_object.append(frame)
                            else:
                                other_objects.append((frame, label.id))

                for a in frames_with_current_object:
                    for b in frames_with_current_object:

                        other_object_frame, other_object_id = other_objects[np.random.randint(
                            0, len(other_objects)
                        )]

                        element = (
                            track_id,
                            a,
                            b,
                            object_id,
                            other_object_frame,
                            other_object_id,
                        )
                        self.track_target_search_frames_with_id_other_frame_id.append(
                            element
                        )

        print()

    def __getitem__(self, idx):

        (
            track_id,
            target_frame_id,
            search_frame_id,
            object_id,
            other_object_frame,
            other_object_id,
        ) = self.track_target_search_frames_with_id_other_frame_id[idx]

        target_points = np.fromfile(
            os.path.join(
                self.lidar_paths[track_id], self.lidar_files[track_id][target_frame_id]
            ),
            dtype=np.float32,
            count=-1,
        ).reshape([-1, self.num_point_features])

        search_points = np.fromfile(
            os.path.join(
                self.lidar_paths[track_id], self.lidar_files[track_id][search_frame_id]
            ),
            dtype=np.float32,
            count=-1,
        ).reshape([-1, self.num_point_features])

        other_points = np.fromfile(
            os.path.join(
                self.lidar_paths[track_id],
                self.lidar_files[track_id][other_object_frame],
            ),
            dtype=np.float32,
            count=-1,
        ).reshape([-1, self.num_point_features])

        target_label = [
            x for x in self.labels[track_id][target_frame_id] if x.id == object_id
        ][0]
        search_label = [
            x for x in self.labels[track_id][search_frame_id] if x.id == object_id
        ][0]
        other_label = [
            x
            for x in self.labels[track_id][other_object_frame]
            if x.id == other_object_id
        ][0]

        result = (
            PointCloudWithCalibration(target_points, self.calibs[track_id], None),
            PointCloudWithCalibration(search_points, self.calibs[track_id], None),
            PointCloudWithCalibration(other_points, self.calibs[track_id], None),
            target_label,
            search_label,
            other_label,
        )

        return result

    def __len__(self):
        return len(self.track_target_search_frames_with_id_other_frame_id)


def load_tracking_file(file_path, format, return_format, remove_dontcare=False):

    results = {}
    max_frame = -1
    max_id = 0

    with open(file_path) as f:
        lines = [x.strip() for x in f.readlines()]

    for line in lines:
        fields = line.split(" ")
        frame = int(float(fields[0]))

        if format == "tracking":
            if return_format == "tracking":
                box = TrackingAnnotation3D(
                    name=fields[2],
                    truncated=int(float(fields[3])),
                    occluded=int(float(fields[4])),
                    alpha=float(fields[5]),
                    bbox2d=[
                        float(fields[6]),
                        float(fields[7]),
                        float(fields[8]),
                        float(fields[9]),
                    ],
                    dimensions=[
                        float(fields[12]),
                        float(fields[10]),
                        float(fields[11]),
                    ],
                    location=[float(fields[13]), float(fields[14]), float(fields[15])],
                    rotation_y=float(fields[16]),
                    score=0 if len(fields) <= 17 else fields[17],
                    frame=int(float(fields[0])),
                    id=int(float(fields[1])),
                )
            elif return_format == "detection":
                box = BoundingBox3D(
                    name=fields[2],
                    truncated=int(float(fields[3])),
                    occluded=int(float(fields[4])),
                    alpha=float(fields[5]),
                    bbox2d=[
                        float(fields[6]),
                        float(fields[7]),
                        float(fields[8]),
                        float(fields[9]),
                    ],
                    dimensions=[
                        float(fields[12]),
                        float(fields[10]),
                        float(fields[11]),
                    ],
                    location=[float(fields[13]), float(fields[14]), float(fields[15])],
                    rotation_y=float(fields[16]),
                    score=0 if len(fields) <= 17 else fields[17],
                )
            else:
                raise ValueError("return_format should be tracking or detection")
        elif format == "detection":
            if return_format == "tracking":
                box = TrackingAnnotation3D(
                    name=fields[1],
                    truncated=int(float(fields[2])),
                    occluded=int(float(fields[3])),
                    alpha=float(fields[4]),
                    bbox2d=[
                        float(fields[5]),
                        float(fields[6]),
                        float(fields[7]),
                        float(fields[8]),
                    ],
                    dimensions=[
                        float(fields[11]),
                        float(fields[9]),
                        float(fields[10]),
                    ],
                    location=[float(fields[12]), float(fields[13]), float(fields[14])],
                    rotation_y=float(fields[15]),
                    score=0 if len(fields) <= 15 else fields[16],
                    frame=int(float(fields[0])),
                    id=-1,
                )
            elif return_format == "detection":
                box = BoundingBox3D(
                    name=fields[1],
                    truncated=int(float(fields[2])),
                    occluded=int(float(fields[3])),
                    alpha=float(fields[4]),
                    bbox2d=[
                        float(fields[5]),
                        float(fields[6]),
                        float(fields[7]),
                        float(fields[8]),
                    ],
                    dimensions=[
                        float(fields[11]),
                        float(fields[9]),
                        float(fields[10]),
                    ],
                    location=[float(fields[12]), float(fields[13]), float(fields[14])],
                    rotation_y=float(fields[15]),
                    score=0 if len(fields) <= 15 else fields[16],
                )
        else:
            raise ValueError("format should be tracking or detection")

        if frame not in results:
            results[frame] = []

        if not (remove_dontcare and box.name == "DontCare"):
            results[frame].append(box)
            max_frame = max(max_frame, frame)
            max_id = max(max_id, box.id)

    if return_format == "tracking":

        result = []

        for frame in range(max_frame):
            if frame in results:
                result.append(TrackingAnnotation3DList(results[frame]))
            else:
                result.append(TrackingAnnotation3DList([]))

        return result, max_id
    elif return_format == "detection":
        result = []

        for frame in range(max_frame):
            if frame in results:
                result.append(BoundingBox3DList(results[frame]))
            else:
                result.append(BoundingBox3DList([]))

        return result, max_id
    else:
        raise ValueError("return_format should be tracking or detection")
