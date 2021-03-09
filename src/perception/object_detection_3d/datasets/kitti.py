
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

import os
import numpy as np
from engine.datasets import ExternalDataset, DatasetIterator
from engine.data import PointCloudWithCalibration
from engine.target import BoundingBox3DList
from perception.object_detection_3d.datasets.create_data_kitti import (
    create_kitti_info_file,
    create_reduced_point_cloud,
    create_groundtruth_database,
)
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.data.kitti_common import (
     get_label_anno, _extend_matrix
)


class DatasetSamplerOptions:
    def __init__(self):

        super().__init__()


class KittiDataset(ExternalDataset):
    def __init__(self, path, kitti_subsets_path="./perception/object_detection_3d/datasets/kitti_subsets"):

        super().__init__(path, "kitti")

        self.path = path
        self.kitti_subsets_path = kitti_subsets_path

        self.__prepare_data()

    def __prepare_data(self):

        files = os.listdir(self.path)

        if ("gt_database" in files) and ("kitti_infos_train.pkl" in files):
            print(":::Data Ready:::")
            return

        print(":::Create KITTI Info File:::")
        create_kitti_info_file(self.path, self.kitti_subsets_path)

        print(":::Create Reduced Point Cloud:::")
        create_reduced_point_cloud(self.path)

        print(":::Create Ground-Truth Database:::")
        create_groundtruth_database(self.path)

        print(":::Data Ready:::")

        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return super().__len__()


class LabeledPointCloudsDatasetIterator(DatasetIterator):
    def __init__(self, lidar_path, label_path, calib_path, num_point_features=4):
        super().__init__()

        self.lidar_path = lidar_path
        self.label_path = label_path
        self.calib_path = calib_path
        self.num_point_features = num_point_features

        self.lidar_files = os.listdir(self.lidar_path)
        self.label_files = os.listdir(self.label_path)
        self.calib_files = os.listdir(self.calib_path)

        if (
            len(self.lidar_files) != len(self.label_files) or
            len(self.lidar_files) != len(self.calib_files)
        ):
            raise ValueError("Number of files in lidar, label and calib files is not identical")

    def __getitem__(self, idx):
        points = np.fromfile(
            os.path.join(self.lidar_path, self.lidar_files[idx]), dtype=np.float32, count=-1
        ).reshape([-1, self.num_point_features])
        calib = parse_calib(
            os.path.join(self.calib_path, self.calib_files[idx])
        )
        target = BoundingBox3DList.from_kitti(get_label_anno(
            os.path.join(self.label_path, self.label_files[idx])
        ))

        result = (
            PointCloudWithCalibration(points, calib),
            target
        )

        return result

    def __len__(self):
        return len(self.lidar_files)


def parse_calib(
    calib_path,
    extend_matrix=True,
):

    result = {}

    with open(calib_path, "r") as f:
        lines = f.readlines()
    P0 = np.array([float(info) for info in lines[0].split(" ")[1:13]
                ]).reshape([3, 4])
    P1 = np.array([float(info) for info in lines[1].split(" ")[1:13]
                ]).reshape([3, 4])
    P2 = np.array([float(info) for info in lines[2].split(" ")[1:13]
                ]).reshape([3, 4])
    P3 = np.array([float(info) for info in lines[3].split(" ")[1:13]
                ]).reshape([3, 4])
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
    result["P0"] = P0
    result["P1"] = P1
    result["P2"] = P2
    result["P3"] = P3
    R0_rect = np.array([
        float(info) for info in lines[4].split(" ")[1:10]
    ]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.0
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect
    result["R0_rect"] = rect_4x4
    Tr_velo_to_cam = np.array([
        float(info) for info in lines[5].split(" ")[1:13]
    ]).reshape([3, 4])
    Tr_imu_to_velo = np.array([
        float(info) for info in lines[6].split(" ")[1:13]
    ]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    result["Tr_velo_to_cam"] = Tr_velo_to_cam
    result["Tr_imu_to_velo"] = Tr_imu_to_velo

    return result
