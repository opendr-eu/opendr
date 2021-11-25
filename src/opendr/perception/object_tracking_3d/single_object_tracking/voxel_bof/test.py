import sys
import os
import torch
from opendr.engine.target import TrackingAnnotation3D, TrackingAnnotation3DList
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.voxel_bof_object_tracking_3d_learner import (
    VoxelBofObjectTracking3DLearner,
)
from opendr.perception.object_detection_3d.datasets.kitti import (
    KittiDataset,
    LabeledPointCloudsDatasetIterator,
)
from opendr.perception.object_tracking_3d.datasets.kitti_tracking import (
    KittiTrackingDatasetIterator,
    LabeledTrackingPointCloudsDatasetIterator,
)
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.draw import (
    draw_point_cloud_bev,
    draw_point_cloud_projected_2,
)
from PIL import Image as PilImage
import numpy as np
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.core.box_np_ops import (
    box_camera_to_lidar,
    camera_to_lidar,
    center_to_corner_box3d,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)
print("Using device:", DEVICE, file=sys.stderr)

dataset_detection_path = "/data/sets/kitti_second"
dataset_tracking_path = "/data/sets/kitti_tracking"

temp_dir = os.path.join(
    "tests",
    "sources",
    "tools",
    "perception",
    "object_detection_3d",
    "voxel_object_detection_3d",
    "voxel_object_detection_3d_temp",
)

config_tanet_car = "src/opendr/perception/object_tracking_3d/single_object_tracking/voxel_bof/second_detector/configs/tanet/car/xyres_16.proto"
config_tanet_ped_cycle = "src/opendr/perception/object_tracking_3d/single_object_tracking/voxel_bof/second_detector/configs/tanet/ped_cycle/xyres_16.proto"
config_pointpillars_car = "src/opendr/perception/object_tracking_3d/single_object_tracking/voxel_bof/second_detector/configs/pointpillars/car/xyres_16.proto"
config_pointpillars_ped_cycle = "src/opendr/perception/object_tracking_3d/single_object_tracking/voxel_bof/second_detector/configs/pointpillars/ped_cycle/xyres_16.proto"

subsets_path = os.path.join(
    ".",
    "src",
    "opendr",
    "perception",
    "object_detection_3d",
    "datasets",
    "kitti_subsets",
)

model_paths = {
    "tanet_car": "models/tanet_car_xyres_16",
    "tanet_ped_cycle": "models/tanet_ped_cycle_xyres_16",
    "pointpillars_car": "models/pointpillars_car_xyres_16",
    "pointpillars_ped_cycle": "models/pointpillars_ped_cycle_xyres_16",
}

all_configs = {
    "tanet_car": config_tanet_car,
    "tanet_ped_cycle": config_tanet_ped_cycle,
    "pointpillars_car": config_pointpillars_car,
    "pointpillars_ped_cycle": config_pointpillars_ped_cycle,
}
car_configs = {
    "tanet_car": config_tanet_car,
    "pointpillars_car": config_pointpillars_car,
}

kitti_detection = KittiDataset(dataset_detection_path)
dataset_detection = LabeledPointCloudsDatasetIterator(
    dataset_detection_path + "/training/velodyne_reduced",
    dataset_detection_path + "/training/label_2",
    dataset_detection_path + "/training/calib",
)
track_id = "0002"
dataset_tracking = LabeledTrackingPointCloudsDatasetIterator(
    dataset_tracking_path + "/training/velodyne/" + track_id,
    dataset_tracking_path + "/training/label_02/" + track_id + ".txt",
    dataset_tracking_path + "/training/calib/" + track_id + ".txt",
)
name = "pointpillars_car"
config = all_configs[name]
model_path = model_paths[name]


tanet_name = "tanet_car"
tanet_config = all_configs[tanet_name]
tanet_model_path = model_paths[tanet_name]


pq = 20
lq = 10


def tracking_boxes_to_lidar(
    label, calib, classes=["Car", "Van", "Pedestrian", "Cyclist"]
):

    label = label.kitti()

    r0_rect = calib["R0_rect"]
    trv2c = calib["Tr_velo_to_cam"]

    label_to_id = {
        "Car": 0,
        "Van": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
    }

    background_id = -1

    class_ids = [
        (
            label_to_id[name]
            if (name in label_to_id and name in classes)
            else background_id
        )
        for name in label["name"]
    ]

    selected_objects = []

    for i, class_id in enumerate(class_ids):
        if class_id != background_id:
            selected_objects.append(i)

    dims = label["dimensions"][selected_objects]
    locs = label["location"][selected_objects]
    rots = label["rotation_y"][selected_objects]

    gt_boxes_camera = np.concatenate(
        [locs, dims, rots[..., np.newaxis]], axis=1
    )
    gt_boxes_lidar = box_camera_to_lidar(gt_boxes_camera, r0_rect, trv2c)
    locs_lidar = gt_boxes_lidar[:, 0:3]
    dims_lidar = gt_boxes_lidar[:, 3:6]
    rots_lidar = gt_boxes_lidar[:, 6:7]

    new_label = {
        "name": label["name"][selected_objects],
        "truncated": label["truncated"][selected_objects],
        "occluded": label["occluded"][selected_objects],
        "alpha": label["alpha"][selected_objects],
        "bbox": label["bbox"][selected_objects],
        "dimensions": dims_lidar,
        "location": locs_lidar,
        "rotation_y": rots_lidar,
        "score": label["score"][selected_objects],
        "id": label["id"][selected_objects]
        if "id" in label
        else np.array(list(range(len(selected_objects)))),
        "frame": label["frame"][selected_objects]
        if "frame" in label
        else np.array([0] * len(selected_objects)),
    }

    result = TrackingAnnotation3DList.from_kitti(
        new_label, new_label["id"], new_label["frame"]
    )

    return result


def label_to_AABB(label):

    if len(label) == 0:
        return label

    label = label.kitti()

    dims = label["dimensions"]
    locs = label["location"]
    rots = label["rotation_y"]

    origin = [0.5, 0.5, 0]
    gt_corners = center_to_corner_box3d(
        locs, dims, rots.reshape(-1), origin=origin, axis=2,
    )

    mins = np.min(gt_corners, axis=1)
    maxs = np.max(gt_corners, axis=1)
    centers = (maxs + mins) / 2
    sizes = maxs - mins
    rotations = np.zeros((centers.shape[0],), dtype=np.float32)

    new_label = {
        "name": label["name"],
        "truncated": label["truncated"],
        "occluded": label["occluded"],
        "alpha": label["alpha"],
        "bbox": label["bbox"],
        "dimensions": sizes,
        "location": centers,
        "rotation_y": rotations,
        "score": label["score"],
        "id": label["id"],
        "frame": label["frame"],
    }

    result = TrackingAnnotation3DList.from_kitti(
        new_label, new_label["id"], new_label["frame"]
    )

    return result


def test_eval_detection():
    print("Eval", name, "start", file=sys.stderr)
    model_path = model_paths[name]

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config, device=DEVICE
    )
    learner.load(model_path)
    mAPbbox, mAPbev, mAP3d, mAPaos = learner.eval(dataset_detection)

    print(
        "Ok?", mAPbbox[0][0][0] > 70 and mAPbbox[0][0][0] < 95,
    )


def test_draw_tracking_dataset():

    for q in range(lq):  # range(len(dataset_tracking)):\
        i = q * pq
        print(i, "/", len(dataset_tracking))
        point_cloud_with_calibration, label = dataset_tracking[i]
        point_cloud = point_cloud_with_calibration.data
        calib = point_cloud_with_calibration.calib
        lidar_boxes = tracking_boxes_to_lidar(label, calib)
        image = draw_point_cloud_bev(point_cloud, lidar_boxes)
        PilImage.fromarray(image).save("./plots/kt_" + str(i) + ".png")


def test_draw_detection_dataset():

    for q in range(lq):  # range(len(dataset_tracking)):\
        i = q
        print(i, "/", len(dataset_detection))
        point_cloud_with_calibration, label = dataset_detection[i]
        point_cloud = point_cloud_with_calibration.data
        calib = point_cloud_with_calibration.calib
        lidar_boxes = tracking_boxes_to_lidar(label, calib)
        image = draw_point_cloud_bev(point_cloud, lidar_boxes)
        PilImage.fromarray(image).save("./plots/kd_" + str(i) + ".png")


def test_draw_detection_projected():

    for q in range(lq):  # range(len(dataset_tracking)):\
        i = q
        print(i, "/", len(dataset_detection))
        point_cloud_with_calibration, label = dataset_detection[i]
        point_cloud = point_cloud_with_calibration.data
        calib = point_cloud_with_calibration.calib
        lidar_boxes = tracking_boxes_to_lidar(label, calib)
        image = draw_point_cloud_projected_2(point_cloud, lidar_boxes)
        PilImage.fromarray(image).save("./plots/dpr_" + str(i) + ".png")


def test_draw_tracking_projected():

    for q in range(lq):  # range(len(dataset_tracking)):\
        i = q * pq
        print(i, "/", len(dataset_tracking))
        point_cloud_with_calibration, label = dataset_tracking[i]
        point_cloud = point_cloud_with_calibration.data
        calib = point_cloud_with_calibration.calib
        lidar_boxes = tracking_boxes_to_lidar(label, calib)
        image = draw_point_cloud_projected_2(point_cloud, lidar_boxes)
        PilImage.fromarray(image).save("./plots/pr_" + str(i) + ".png")


def test_draw_tracking_aabb():

    for q in range(lq):  # range(len(dataset_tracking)):\
        i = q * pq
        print(i, "/", len(dataset_tracking))
        point_cloud_with_calibration, label = dataset_tracking[i]
        point_cloud = point_cloud_with_calibration.data
        calib = point_cloud_with_calibration.calib
        lidar_boxes = tracking_boxes_to_lidar(label, calib)
        aabb = label_to_AABB(lidar_boxes)
        image = draw_point_cloud_bev(point_cloud, aabb)
        PilImage.fromarray(image).save("./plots/aabb_" + str(i) + ".png")


def test_pp_infer_tracking():
    print("Eval", name, "start", file=sys.stderr)

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config, device=DEVICE
    )
    learner.load(model_path)

    for q in range(lq):  # range(len(dataset_tracking)):\
        i = q * pq
        print(i, "/", len(dataset_tracking))
        point_cloud_with_calibration, label = dataset_tracking[i]
        predictions = learner.infer(point_cloud_with_calibration)
        image = draw_point_cloud_bev(
            point_cloud_with_calibration.data, predictions
        )
        PilImage.fromarray(image).save("./plots/pp_" + str(i) + ".png")


def test_pp_infer_detection():
    print("Eval", name, "start", file=sys.stderr)

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config, device=DEVICE
    )
    learner.load(model_path)

    for q in range(lq):  # range(len(dataset_tracking)):\
        i = q * pq
        print(i, "/", len(dataset_tracking))
        point_cloud_with_calibration, label = dataset_detection[i]
        predictions = learner.infer(point_cloud_with_calibration)
        image = draw_point_cloud_bev(
            point_cloud_with_calibration.data, predictions
        )
        PilImage.fromarray(image).save("./plots/dpp_" + str(i) + ".png")


def test_tanet_infer_tracking():
    print("Eval", tanet_name, "start", file=sys.stderr)

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=tanet_config, device=DEVICE
    )

    if not os.path.exists(tanet_model_path):
        learner.download("tanet_car_xyres_16", "models")

    learner.load(tanet_model_path)

    for q in range(lq):  # range(len(dataset_tracking)):\
        i = q * pq
        print(i, "/", len(dataset_tracking))
        point_cloud_with_calibration, label = dataset_tracking[i]
        predictions = learner.infer(point_cloud_with_calibration)
        image = draw_point_cloud_bev(
            point_cloud_with_calibration.data, predictions
        )
        PilImage.fromarray(image).save("./plots/tanet_" + str(i) + ".png")


def test_pp_siamese():
    print("Eval", name, "start", file=sys.stderr)

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config, device=DEVICE
    )
    learner.load(model_path)
    learner.fit(
        kitti_detection, 
        model_dir="./temp/0",
        verbose=True
    )

    print()


test_pp_siamese()

# test_tanet_infer_tracking()
# test_pp_infer_tracking()
# test_pp_infer_detection()

# test_eval_detection()

# test_draw_tracking_projected()
# test_draw_tracking_dataset()
# test_draw_detection_dataset()
# test_draw_detection_projected()
# test_draw_tracking_aabb()
