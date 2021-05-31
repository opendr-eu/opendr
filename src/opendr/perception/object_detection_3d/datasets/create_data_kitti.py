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
# The functions are taken from TANet code

import pathlib
import pickle

import numpy as np

from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.core import (
    box_np_ops,
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.data import (
    kitti_common as kitti,
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.utils.progress_bar import (
    list_bar as prog_bar,
)
import os


def _read_imageset_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(
    data_path, infos, relative_path, remove_outside=True, num_features=4
):
    for info in infos:
        if relative_path:
            v_path = str(pathlib.Path(data_path) / info["velodyne_path"])
        else:
            v_path = info["velodyne_path"]
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape(
            [-1, num_features]
        )
        rect = info["calib/R0_rect"]
        Trv2c = info["calib/Tr_velo_to_cam"]
        P2 = info["calib/P2"]
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, info["img_shape"]
            )

        annos = info["annos"]
        num_obj = len([n for n in annos["name"] if n != "DontCare"])
        dims = annos["dimensions"][:num_obj]
        loc = annos["location"][:num_obj]
        rots = annos["rotation_y"][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos["dimensions"]) - num_obj
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)


def create_kitti_info_file(
    data_path, kitti_subsets_path, save_path=None, create_trainval=False, relative_path=True
):

    kitti_subsets_path = pathlib.Path(kitti_subsets_path)

    train_img_ids = _read_imageset_file(
        kitti_subsets_path / "train.txt"
    )
    val_img_ids = _read_imageset_file(
        kitti_subsets_path / "val.txt"
    )
    test_img_ids = _read_imageset_file(
       kitti_subsets_path / "test.txt"
    )

    print("Generate info. This may take several minutes.")
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)
    kitti_infos_train = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
    )
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / "kitti_infos_train.pkl"
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(kitti_infos_train, f)
    kitti_infos_val = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path,
    )
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / "kitti_infos_val.pkl"
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(kitti_infos_val, f)
    filename = save_path / "kitti_infos_trainval.pkl"
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    kitti_infos_test = kitti.get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path,
    )
    filename = save_path / "kitti_infos_test.pkl"
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(kitti_infos_test, f)


def _create_reduced_point_cloud(data_path, info_path, save_path=None, back=False):
    with open(info_path, "rb") as f:
        kitti_infos = pickle.load(f)
    for info in prog_bar(kitti_infos):
        v_path = info["velodyne_path"]
        v_path = pathlib.Path(data_path) / v_path
        points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
        rect = info["calib/R0_rect"]
        P2 = info["calib/P2"]
        Trv2c = info["calib/Tr_velo_to_cam"]
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(
            points_v, rect, Trv2c, P2, info["img_shape"]
        )

        if save_path is None:
            save_filename = (
                v_path.parent.parent / (v_path.parent.stem + "_reduced") / v_path.name
            )

            os.makedirs(v_path.parent.parent / (v_path.parent.stem + "_reduced"), exist_ok=True)

            if back:
                save_filename += "_back"
        else:
            save_filename = str(pathlib.Path(save_path) / v_path.name)
            if back:
                save_filename += "_back"
        with open(save_filename, "w") as f:
            points_v.tofile(f)


def create_reduced_point_cloud(
    data_path,
    train_info_path=None,
    val_info_path=None,
    test_info_path=None,
    save_path=None,
    with_back=False,
):
    if train_info_path is None:
        train_info_path = pathlib.Path(data_path) / "kitti_infos_train.pkl"
    if val_info_path is None:
        val_info_path = pathlib.Path(data_path) / "kitti_infos_val.pkl"
    if test_info_path is None:
        test_info_path = pathlib.Path(data_path) / "kitti_infos_test.pkl"

    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(data_path, test_info_path, save_path, back=True)


def create_groundtruth_database(
    data_path,
    info_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
    relative_path=True,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
):
    root_path = pathlib.Path(data_path)
    if info_path is None:
        info_path = root_path / "kitti_infos_train.pkl"
    if database_save_path is None:
        database_save_path = root_path / "gt_database"
    else:
        database_save_path = pathlib.Path(database_save_path)
    if db_info_save_path is None:
        db_info_save_path = root_path / "kitti_dbinfos_train.pkl"
    database_save_path.mkdir(parents=True, exist_ok=True)
    with open(info_path, "rb") as f:
        kitti_infos = pickle.load(f)
    all_db_infos = {}
    if used_classes is None:
        used_classes = list(kitti.get_classes())
        used_classes.pop(used_classes.index("DontCare"))
    for name in used_classes:
        all_db_infos[name] = []
    group_counter = 0
    for info in prog_bar(kitti_infos):
        velodyne_path = info["velodyne_path"]
        if relative_path:
            velodyne_path = str(root_path / velodyne_path)
        num_features = 4
        if "pointcloud_num_features" in info:
            num_features = info["pointcloud_num_features"]
        points = np.fromfile(velodyne_path, dtype=np.float32, count=-1).reshape(
            [-1, num_features]
        )

        image_idx = info["image_idx"]
        rect = info["calib/R0_rect"]
        P2 = info["calib/P2"]
        Trv2c = info["calib/Tr_velo_to_cam"]
        if not lidar_only:
            points = box_np_ops.remove_outside_points(
                points, rect, Trv2c, P2, info["img_shape"]
            )

        annos = info["annos"]
        names = annos["name"]
        bboxes = annos["bbox"]
        difficulty = annos["difficulty"]
        gt_idxes = annos["index"]
        num_obj = np.sum(annos["index"] >= 0)
        rbbox_cam = kitti.anno_to_rbboxes(annos)[:num_obj]
        rbbox_lidar = box_np_ops.box_camera_to_lidar(rbbox_cam, rect, Trv2c)
        if bev_only:  # set z and h to limits
            assert coors_range is not None
            rbbox_lidar[:, 2] = coors_range[2]
            rbbox_lidar[:, 5] = coors_range[5] - coors_range[2]

        group_dict = {}
        group_ids = np.full([bboxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(bboxes.shape[0], dtype=np.int64)
        point_indices = box_np_ops.points_in_rbbox(points, rbbox_lidar)
        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{gt_idxes[i]}.bin"
            filepath = database_save_path / filename
            gt_points = points[point_indices[:, i]]

            gt_points[:, :3] -= rbbox_lidar[i, :3]
            with open(filepath, "w") as f:
                gt_points.tofile(f)
            if names[i] in used_classes:
                if relative_path:
                    db_path = str(database_save_path.stem + "/" + filename)
                else:
                    db_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": gt_idxes[i],
                    "box3d_lidar": rbbox_lidar[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }

                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                all_db_infos[names[i]].append(db_info)
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)
