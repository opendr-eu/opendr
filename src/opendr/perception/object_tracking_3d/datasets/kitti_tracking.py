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

# KITTI tracking data format:
# (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry,[score])
# KITTI detection data format (with frame):
# (frame,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry,[score])

import os
import time
from zipfile import ZipFile
from urllib.request import urlretrieve
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.target import (
    BoundingBox3D,
    BoundingBox3DList,
    TrackingAnnotation3D,
    TrackingAnnotation3DList,
)
from opendr.engine.data import PointCloudWithCalibration
from opendr.engine.datasets import DatasetIterator
import numpy as np
from skimage import io

from opendr.perception.object_detection_3d.datasets.kitti import parse_calib


class KittiTrackingDatasetIterator(DatasetIterator):
    def __init__(
        self, inputs_path, ground_truths_path,
        inputs_format="detection"  # detection, tracking
    ):
        super().__init__()

        self.inputs_path = inputs_path
        self.ground_truths_path = ground_truths_path
        self.inputs_format = inputs_format

        self.inputs_files = os.listdir(self.inputs_path)
        self.ground_truths_files = os.listdir(self.ground_truths_path)

        if len(self.inputs_files) != len(self.ground_truths_files):
            raise ValueError(
                "Number of input files and ground_truth files is not identical"
            )

        self.__load_data()

    @staticmethod
    def download(
        url, download_path, inputs_sub_path=".", ground_truths_sub_path=".",
        inputs_format="detection", file_format="zip",
        create_dir=False
    ):

        if file_format == "zip":
            if create_dir:
                os.makedirs(download_path, exist_ok=True)

            print("Downloading KITTI Tracking detections + labels dataset zip file from", url, "to", download_path)

            start_time = 0
            last_print = 0

            def reporthook(count, block_size, total_size):
                nonlocal start_time
                nonlocal last_print
                if count == 0:
                    start_time = time.time()
                    last_print = start_time
                    return

                duration = time.time() - start_time
                progress_size = int(count * block_size)
                speed = int(progress_size / (1024 * duration))
                if time.time() - last_print >= 1:
                    last_print = time.time()
                    print(
                        "\r%d MB, %d KB/s, %d seconds passed" %
                        (progress_size / (1024 * 1024), speed, duration),
                        end=''
                    )

            zip_path = os.path.join(download_path, "dataset.zip")
            urlretrieve(url, zip_path, reporthook=reporthook)
            print()

            print("Extracting KITTI Dataset from zip file")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)

            os.remove(zip_path)

            return KittiTrackingDatasetIterator(
                os.path.join(download_path, inputs_sub_path),
                os.path.join(download_path, ground_truths_sub_path),
                inputs_format
            )

        else:
            raise ValueError("Unsupported file_format: " + file_format)

    @staticmethod
    def download_labels(
        download_path, create_dir=False,
    ):
        return KittiTrackingDatasetIterator.download(
            os.path.join(OPENDR_SERVER_URL, "perception", "object_tracking_3d", "kitti_tracking_labels.zip"),
            download_path,
            "kitti_tracking_labels",
            "kitti_tracking_labels",
            "tracking",
            create_dir=create_dir,
        )

    def __getitem__(self, idx):

        return self.data[idx]

    def __load_data(self):

        data = []

        for input_file, ground_truth_file in zip(
            self.inputs_files,
            self.ground_truths_files
        ):

            input, _ = load_tracking_file(
                os.path.join(self.inputs_path, input_file), self.inputs_format, "detection", remove_dontcare=True
            )
            ground_truth, _ = load_tracking_file(
                os.path.join(self.ground_truths_path, ground_truth_file), "tracking", "tracking"
            )

            data.append((input, ground_truth))

        self.data = data

    def __len__(self):
        return len(self.data)


class LabeledTrackingPointCloudsDatasetIterator(DatasetIterator):
    def __init__(
        self,
        lidar_path,
        label_path,
        calib_path,
        image_path=None,
        labels_format="tracking",  # detection, tracking
        num_point_features=4,
    ):
        super().__init__()

        self.lidar_path = lidar_path
        self.label_path = label_path
        self.calib_path = calib_path
        self.image_path = image_path
        self.num_point_features = num_point_features

        self.lidar_files = sorted(os.listdir(self.lidar_path))
        # self.label_files = sorted(os.listdir(self.label_path))
        # self.calib_files = sorted(os.listdir(self.calib_path))
        self.image_files = (
            sorted(os.listdir(self.image_path))
            if self.image_path is not None
            else None
        )

        self.labels, self.max_id = load_tracking_file(
            self.label_path, "tracking", labels_format,
        )
        self.calib = parse_calib(self.calib_path)

    def __getitem__(self, idx):
        points = np.fromfile(
            os.path.join(self.lidar_path, self.lidar_files[idx]),
            dtype=np.float32,
            count=-1,
        ).reshape([-1, self.num_point_features])
        target = self.labels[idx] if len(self.labels) > idx else TrackingAnnotation3DList([])

        image_shape = (
            None
            if self.image_files is None
            else (
                np.array(
                    io.imread(
                        os.path.join(self.image_path, self.image_files[idx])
                    ).shape[:2],
                    dtype=np.int32,
                )
            )
        )

        result = (
            PointCloudWithCalibration(points, self.calib, image_shape),
            target,
        )

        return result

    def __len__(self):
        return len(self.lidar_files)


def load_tracking_file(
    file_path, format, return_format, remove_dontcare=False
):

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
                    location=[
                        float(fields[13]),
                        float(fields[14]),
                        float(fields[15]),
                    ],
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
                    location=[
                        float(fields[13]),
                        float(fields[14]),
                        float(fields[15]),
                    ],
                    rotation_y=float(fields[16]),
                    score=0 if len(fields) <= 17 else fields[17],
                )
            else:
                raise ValueError(
                    "return_format should be tracking or detection"
                )
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
                    location=[
                        float(fields[12]),
                        float(fields[13]),
                        float(fields[14]),
                    ],
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
                    location=[
                        float(fields[12]),
                        float(fields[13]),
                        float(fields[14]),
                    ],
                    rotation_y=float(fields[15]),
                    score=0 if len(fields) <= 15 else fields[16],
                )
        else:
            raise ValueError("format should be tracking or detection")

        if frame not in results:
            results[frame] = []
            max_frame = max(max_frame, frame)

        if not (remove_dontcare and box.name == "DontCare"):
            results[frame].append(box)

            if isinstance(box, TrackingAnnotation3D):
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
