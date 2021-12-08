import sys
import os
import torch
from opendr.engine.target import TrackingAnnotation3D, TrackingAnnotation3DList
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.utils.eval import d3_box_overlap
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.run import (
    iou_2d,
)
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.test import label_to_AABB, tracking_boxes_to_camera, tracking_boxes_to_lidar
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
track_id = "0000"
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


pq = 1
lq = 20


def test_rotated_pp_siamese_eval(draw=True, iou_min=0.0, classes=["Car", "Van", "Truck"]):
    print("Eval", name, "start", file=sys.stderr)
    import pygifsicle
    import imageio

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config,
        device=DEVICE,
        lr=0.001,
        checkpoint_after_iter=2000,
    )
    # learner.load(model_path, backbone=True, verbose=True)
    learner.load("./temp/upscaled-rotated-0/checkpoints", backbone=False, verbose=True)

    def test_track(track_id):
        count = len(dataset_tracking)
        # count = 120
        dataset = LabeledTrackingPointCloudsDatasetIterator(
            dataset_tracking_path + "/training/velodyne/" + track_id,
            dataset_tracking_path + "/training/label_02/" + track_id + ".txt",
            dataset_tracking_path + "/training/calib/" + track_id + ".txt",
        )

        all_mean_iou3ds = []
        all_mean_iouAabbs = []
        all_tracked = []

        def test_object_id(object_id):

            start_frame = -1

            selected_labels = []

            while len(selected_labels) <= 0:
                start_frame += 1
                point_cloud_with_calibration, labels = dataset[start_frame]
                selected_labels = TrackingAnnotation3DList(
                    [label for label in labels if (label.id == object_id)]
                )

            if not selected_labels[0].name in classes:
                return None, None, None

            calib = point_cloud_with_calibration.calib
            labels_lidar = tracking_boxes_to_lidar(selected_labels, calib, classes=classes)
            label_lidar = labels_lidar[0]

            learner.init(point_cloud_with_calibration, label_lidar)

            images = []
            ious = []
            count_tracked = 0

            for i in range(start_frame, count):
                point_cloud_with_calibration, labels = dataset[i]
                selected_labels = TrackingAnnotation3DList(
                    [label for label in labels if label.id == object_id]
                )

                if len(selected_labels) <= 0:
                    break

                calib = point_cloud_with_calibration.calib
                labels_lidar = tracking_boxes_to_lidar(selected_labels, calib)
                label_lidar = (
                    labels_lidar[0] if len(labels_lidar) > 0 else None
                )

                result = learner.infer(
                    point_cloud_with_calibration, id=-1, frame=i, draw=False,
                )

                all_labels = (
                    result
                    if label_lidar is None
                    else TrackingAnnotation3DList([result[0], label_lidar])
                )
                image = draw_point_cloud_bev(
                    point_cloud_with_calibration.data, all_labels
                )

                if draw:
                    pil_image = PilImage.fromarray(image)
                    images.append(pil_image)

                iouAabb = iou_2d(
                    result[0].location[:2],
                    result[0].dimensions[:2],
                    label_lidar.location[:2],
                    label_lidar.dimensions[:2],
                )

                result = tracking_boxes_to_camera(result, calib)[0]
                label_lidar = tracking_boxes_to_camera(
                    TrackingAnnotation3DList([label_lidar]), calib
                )[0]

                dt_boxes = np.concatenate(
                    [
                        result.location.reshape(1, 3),
                        result.dimensions.reshape(1, 3),
                        result.rotation_y.reshape(1, 1),
                    ],
                    axis=1,
                )
                gt_boxes = np.concatenate(
                    [
                        label_lidar.location.reshape(1, 3),
                        label_lidar.dimensions.reshape(1, 3),
                        label_lidar.rotation_y.reshape(1, 1),
                    ],
                    axis=1,
                )
                iou3d = float(d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64))

                if iou3d > iou_min:
                    count_tracked += 1

                ious.append((iou3d, iouAabb))

                print(track_id, "%", object_id, "[", i, "/", count - 1, "] iou3d =", iou3d, "iouAabb =", iouAabb)

                filename = (
                    "./plots/video/eval_aabb_rotated_track_"
                    + str(track_id)
                    + "_obj_"
                    + str(object_id)
                    + ".gif"
                )

            if len(ious) <= 0:
                mean_iou3d = None
                mean_iouAabb = None
                tracked = None
            else:
                mean_iou3d = sum([iou3d for iou3d, iouAabb in ious]) / len(ious)
                mean_iouAabb = sum([iouAabb for iou3d, iouAabb in ious]) / len(ious)
                tracked = count_tracked / len(ious)

            print("mean_iou3d =", mean_iou3d)
            print("mean_iouAabb =", mean_iouAabb)
            print("tracked =", tracked)

            if draw:
                imageio.mimsave(filename, images)
                pygifsicle.optimize(filename)

            return mean_iou3d, mean_iouAabb, tracked

        for object_id in range(0, min(5, dataset.max_id + 1)):
            mean_iou3d, mean_iouAabb, tracked = test_object_id(object_id)

            if mean_iou3d is not None:
                all_mean_iou3ds.append(mean_iou3d)
                all_mean_iouAabbs.append(mean_iouAabb)
                all_tracked.append(tracked)

        if len(all_mean_iou3ds) > 0:
            track_mean_iou3d = sum(all_mean_iou3ds) / len(all_mean_iou3ds)
            track_mean_iouAabb = sum(all_mean_iouAabbs) / len(all_mean_iouAabbs)
            track_mean_tracked = sum(all_tracked) / len(all_tracked)
        else:
            track_mean_iou3d = None
            track_mean_iouAabb = None
            track_mean_tracked = None

        print("track_mean_iou3d =", track_mean_iou3d)
        print("track_mean_iouAabb =", track_mean_iouAabb)
        print("track_mean_tracked =", track_mean_tracked)

        return track_mean_iou3d, track_mean_iouAabb, track_mean_tracked

    tracks = [
        "0000",
        "0001",
        "0002",
        "0003",
        "0004",
        # "0005",
        # "0006",
        # "0007",
        # "0008",
        # "0009",
        # "0010",
        # "0011",
        # "0012",
        # "0013",
        # "0014",
        # "0015",
        # "0016",
        # "0017",
        # "0018",
        # "0019",
        # "0020",
    ]

    all_iou3ds = []
    all_iouAabbs = []
    all_tracked = []

    for track in tracks:
        track_mean_iou3d, track_mean_iouAabb, track_mean_tracked = test_track(track)

        if track_mean_iou3d is not None:
            all_iou3ds.append(track_mean_iou3d)
            all_iouAabbs.append(track_mean_iouAabb)
            all_tracked.append(track_mean_tracked)

    total_mean_iou3d = sum(all_iou3ds) / len(all_iou3ds)
    total_mean_iouAabb = sum(all_iouAabbs) / len(all_iouAabbs)
    total_mean_tracked = sum(all_tracked) / len(all_tracked)

    print("total_mean_iou3d =", total_mean_iou3d)
    print("total_mean_iouAabb =", total_mean_iouAabb)
    print("total_mean_tracked =", total_mean_tracked)

    print("all_iou3ds =", all_iou3ds)
    print("all_iouAabbs =", all_iouAabbs)
    print("all_tracked =", all_tracked)


if __name__ == '__main__':
    test_rotated_pp_siamese_eval()
