import sys
import os
import torch
import fire
from opendr.engine.target import TrackingAnnotation3D, TrackingAnnotation3DList
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.utils.eval import (
    d3_box_overlap,
)
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.run import (
    iou_2d,
)
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
    box_lidar_to_camera,
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


def tracking_boxes_to_lidar(
    label_original,
    calib,
    classes=["Car", "Van", "Pedestrian", "Cyclist", "Truck"],
):

    label = label_original.kitti()

    if len(label["name"]) <= 0:
        return label_original

    r0_rect = calib["R0_rect"]
    trv2c = calib["Tr_velo_to_cam"]

    label_to_id = {
        "Car": 0,
        "Van": 0,
        "Truck": 0,
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


def tracking_boxes_to_camera(
    label_original, calib,
):

    label = label_original.kitti()

    if len(label["name"]) <= 0:
        return label_original

    r0_rect = calib["R0_rect"]
    trv2c = calib["Tr_velo_to_cam"]

    dims = label["dimensions"]
    locs = label["location"]
    rots = label["rotation_y"]

    boxes_lidar = np.concatenate([locs, dims, rots.reshape(-1, 1)], axis=1)
    boxes_camera = box_lidar_to_camera(boxes_lidar, r0_rect, trv2c)
    locs_camera = boxes_camera[:, 0:3]
    dims_camera = boxes_camera[:, 3:6]
    rots_camera = boxes_camera[:, 6:7]

    new_label = {
        "name": label["name"],
        "truncated": label["truncated"],
        "occluded": label["occluded"],
        "alpha": label["alpha"],
        "bbox": label["bbox"],
        "dimensions": dims_camera,
        "location": locs_camera,
        "rotation_y": rots_camera,
        "score": label["score"],
        "id": label["id"]
        if "id" in label
        else np.array(list(range(len(label["name"])))),
        "frame": label["frame"]
        if "frame" in label
        else np.array([0] * len(label["name"])),
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


def test_pp_siamese_fit(model_name, load=0, debug=False):
    print("Fit", name, "start", file=sys.stderr)

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config,
        device=DEVICE,
        lr=0.0001,
        checkpoint_after_iter=500,
        checkpoint_load_iter=load,
    )
    learner.load(model_path, backbone=True, verbose=True)
    learner.fit(
        kitti_detection,
        model_dir="./temp/" + model_name,
        debug=debug,
        # verbose=True
    )

    print()


def test_pp_siamese_load_fit():
    print("Fit", name, "start", file=sys.stderr)

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config,
        device=DEVICE,
        lr=0.0001,
        checkpoint_after_iter=2000,
        # checkpoint_load_iter=42000,
    )
    learner.load("./temp/5-a/checkpoints", backbone=False, verbose=True)
    learner.fit(kitti_detection, model_dir="./temp/load-fit", verbose=True)

    print()


def test_pp_siamese_infer():
    print("Infer", name, "start", file=sys.stderr)
    import pygifsicle
    import imageio

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config,
        device=DEVICE,
        lr=0.001,
        checkpoint_after_iter=2000,
    )
    # learner.load(model_path, backbone=True, verbose=True)
    learner.load("./temp/upscaled-0/checkpoints", backbone=False, verbose=True)

    # count = len(dataset_tracking)
    count = 140
    object_id = 0

    point_cloud_with_calibration, labels = dataset_tracking[0]
    selected_labels = TrackingAnnotation3DList(
        [label for label in labels if label.id == object_id]
    )
    calib = point_cloud_with_calibration.calib
    labels_lidar = label_to_AABB(
        tracking_boxes_to_lidar(selected_labels, calib)
    )
    label_lidar = labels_lidar[0]

    learner.init(point_cloud_with_calibration, label_lidar)

    images = []

    for i in range(1, count):
        point_cloud_with_calibration, labels = dataset_tracking[
            i
        ]  # i iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        selected_labels = TrackingAnnotation3DList(
            [label for label in labels if label.id == object_id]
        )
        calib = point_cloud_with_calibration.calib
        labels_lidar = label_to_AABB(
            tracking_boxes_to_lidar(selected_labels, calib)
        )
        label_lidar = labels_lidar[0] if len(labels_lidar) > 0 else None

        result = learner.infer(point_cloud_with_calibration, id=1, frame=i)

        all_labels = (
            result
            if label_lidar is None
            else TrackingAnnotation3DList([result[0], label_lidar])
        )
        image = draw_point_cloud_bev(
            point_cloud_with_calibration.data, all_labels
        )
        pil_image = PilImage.fromarray(image)
        pil_image.save("./plots/eval_aabb_" + str(i) + ".png")

        images.append(pil_image)

        print("[", i, "/", count, "]", result)

    filename = "./plots/video/eval_aabb_scaled_ms_6k.gif"
    imageio.mimsave(filename, images)
    pygifsicle.optimize(filename)


def test_pp_siamese_eval(
    draw=True, iou_min=0.5, classes=["Car", "Van", "Truck"]
):
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
    learner.load("./temp/upscaled-0/checkpoints", backbone=False, verbose=True)

    def test_track(track_id):
        count = len(dataset_tracking)
        # count = 120
        dataset = LabeledTrackingPointCloudsDatasetIterator(
            dataset_tracking_path + "/training/velodyne/" + track_id,
            dataset_tracking_path + "/training/label_02/" + track_id + ".txt",
            dataset_tracking_path + "/training/calib/" + track_id + ".txt",
        )

        all_mean_ious = []
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
                return None, None

            calib = point_cloud_with_calibration.calib
            labels_lidar = label_to_AABB(
                tracking_boxes_to_lidar(
                    selected_labels, calib, classes=classes
                )
            )
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
                labels_lidar = label_to_AABB(
                    tracking_boxes_to_lidar(selected_labels, calib)
                )
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

                iou = iou_2d(
                    result[0].location[:2],
                    result[0].dimensions[:2],
                    label_lidar.location[:2],
                    label_lidar.dimensions[:2],
                )

                if iou > iou_min:
                    count_tracked += 1

                ious.append(iou)

                print(
                    track_id,
                    "%",
                    object_id,
                    "[",
                    i,
                    "/",
                    count - 1,
                    "] iou =",
                    iou,
                )

                filename = (
                    "./plots/video/eval_aabb_scaled_track_"
                    + str(track_id)
                    + "_obj_"
                    + str(object_id)
                    + ".gif"
                )

            if len(ious) <= 0:
                mean_iou = None
                tracked = None
            else:
                mean_iou = sum(ious) / len(ious)
                tracked = count_tracked / len(ious)

            print("mean_iou =", mean_iou)
            print("tracked =", tracked)

            if draw:
                imageio.mimsave(filename, images)
                pygifsicle.optimize(filename)

            return mean_iou, tracked

        for object_id in range(0, min(5, dataset.max_id + 1)):
            mean_iou, tracked = test_object_id(object_id)

            if mean_iou is not None:
                all_mean_ious.append(mean_iou)
                all_tracked.append(tracked)

        if len(all_mean_ious) > 0:
            track_mean_iou = sum(all_mean_ious) / len(all_mean_ious)
            track_mean_tracked = sum(all_tracked) / len(all_tracked)
        else:
            track_mean_iou = None
            track_mean_tracked = None

        print("track_mean_iou =", track_mean_iou)
        print("track_mean_tracked =", track_mean_tracked)

        return track_mean_iou, track_mean_tracked

    tracks = [
        "0000",
        "0001",
        "0002",
        "0003",
        "0004",
        "0005",
        "0006",
        "0007",
        "0008",
        "0009",
        "0010",
        "0011",
        "0012",
        "0013",
        "0014",
        "0015",
        "0016",
        "0017",
        "0018",
        "0019",
        "0020",
    ]

    all_ious = []
    all_tracked = []

    for track in tracks:
        track_mean_iou, track_mean_tracked = test_track(track)

        if track_mean_iou is not None:
            all_ious.append(track_mean_iou)
            all_tracked.append(track_mean_tracked)

    total_mean_iou = sum(all_ious) / len(all_ious)
    total_mean_tracked = sum(all_tracked) / len(all_tracked)

    print("total_mean_iou =", total_mean_iou)
    print("total_mean_tracked =", total_mean_tracked)

    print("all_ious =", all_ious)
    print("all_tracked =", all_tracked)


def test_rotated_pp_siamese_infer(model_name, load=0, classes=["Car", "Van", "Truck"], draw=True, iou_min=0.5):
    print("Infer", name, "start", file=sys.stderr)
    import pygifsicle
    import imageio

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config,
        device=DEVICE,
        lr=0.001,
        checkpoint_after_iter=2000,
    )

    checkpoints_path = "./temp/" + model_name + "/checkpoints"

    if load == 0:
        learner.load(
            checkpoints_path, backbone=False, verbose=True
        )
    else:
        learner.load_from_checkpoint(checkpoints_path, load)

    count = len(dataset_tracking)
    object_ids = [0] # [0, 3]
    count = 160
    start_frame = 10
    dataset = LabeledTrackingPointCloudsDatasetIterator(
        dataset_tracking_path + "/training/velodyne/" + track_id,
        dataset_tracking_path + "/training/label_02/" + track_id + ".txt",
        dataset_tracking_path + "/training/calib/" + track_id + ".txt",
    )

    def test_object_id(object_id, start_frame=-1):

        selected_labels = []

        while len(selected_labels) <= 0:
            start_frame += 1
            point_cloud_with_calibration, labels = dataset[start_frame]
            selected_labels = TrackingAnnotation3DList(
                [label for label in labels if (label.id == object_id)]
            )

        if not selected_labels[0].name in classes:
            return None, None

        calib = point_cloud_with_calibration.calib
        labels_lidar = tracking_boxes_to_lidar(
            selected_labels, calib, classes=classes
        )
        label_lidar = labels_lidar[0]

        learner.init(point_cloud_with_calibration, label_lidar, draw=draw)

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
                point_cloud_with_calibration, id=-1, frame=i, draw=draw,
            )

            label_lidar.data["name"] = "Target"

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
            iou = float(d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64))

            # iou = iou_2d(
            #     result.location[:2],
            #     result.dimensions[:2],
            #     label_lidar.location[:2],
            #     label_lidar.dimensions[:2],
            # ) # * min(result[0].rotation_y / label_lidar.rotation_y, label_lidar.rotation_y / result[0].rotation_y)

            # iou = max(0, iou)

            if iou > iou_min:
                count_tracked += 1

            ious.append(iou)

            print(
                track_id,
                "%",
                object_id,
                "[",
                i,
                "/",
                count - 1,
                "] iou =",
                iou,
            )

        if len(ious) <= 0:
            mean_iou = None
            tracked = None
        else:
            mean_iou = sum(ious) / len(ious)
            tracked = count_tracked / len(ious)

        print("mean_iou =", mean_iou)
        print("tracked =", tracked)

        filename = lambda x: (
            "./plots/video/" + x + "_" + model_name + "_track_"
            + str(track_id)
            + "_obj_"
            + str(object_id)
            + "_steps_"
            + str(load)
            + ".gif"
        )

        if draw:
            imageio.mimsave(filename("infer"), images)
            pygifsicle.optimize(filename("infer"))
            print("Saving", "infer", "video")

            for group, images in learner._images.items():
                print("Saving", group, "video")
                imageio.mimsave(filename(group), images)
                pygifsicle.optimize(filename(group))

        return mean_iou, tracked

    for object_id in object_ids:
        test_object_id(object_id, start_frame)


def test_rotated_pp_siamese_eval(
    draw=False, iou_min=0.0, classes=["Car", "Van", "Truck"]
):
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
    learner.load(
        "./temp/no-db-sample-upscaled-rotated-0/checkpoints", backbone=False, verbose=True
    )

    def test_track(track_id):
        count = len(dataset_tracking)
        # count = 120
        dataset = LabeledTrackingPointCloudsDatasetIterator(
            dataset_tracking_path + "/training/velodyne/" + track_id,
            dataset_tracking_path + "/training/label_02/" + track_id + ".txt",
            dataset_tracking_path + "/training/calib/" + track_id + ".txt",
        )

        all_mean_ious = []
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
                return None, None

            calib = point_cloud_with_calibration.calib
            labels_lidar = tracking_boxes_to_lidar(
                selected_labels, calib, classes=classes
            )
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

                result = tracking_boxes_to_camera(result, calib)[0]
                label_lidar = tracking_boxes_to_camera(
                    TrackingAnnotation3DList([label_lidar]), calib
                )[0]

                dt_boxes = np.concatenate(
                    [
                        result[0].location.reshape(1, 3),
                        result[0].dimensions.reshape(1, 3),
                        result[0].rotation_y.reshape(1, 3),
                    ],
                    axis=1,
                )
                gt_boxes = np.concatenate(
                    [
                        label_lidar.location.reshape(1, 3),
                        label_lidar.dimensions.reshape(1, 3),
                        label_lidar.rotation_y.reshape(1, 3),
                    ],
                    axis=1,
                )
                iou = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)

                # iou = iou_2d(
                #     result.location[:2],
                #     result.dimensions[:2],
                #     label_lidar.location[:2],
                #     label_lidar.dimensions[:2],
                # ) # * min(result[0].rotation_y / label_lidar.rotation_y, label_lidar.rotation_y / result[0].rotation_y)

                # iou = max(0, iou)

                if iou > iou_min:
                    count_tracked += 1

                ious.append(iou)

                print(
                    track_id,
                    "%",
                    object_id,
                    "[",
                    i,
                    "/",
                    count - 1,
                    "] iou =",
                    iou,
                )

                filename = (
                    "./plots/video/eval_aabb_scaled_track_"
                    + str(track_id)
                    + "_obj_"
                    + str(object_id)
                    + ".gif"
                )

            if len(ious) <= 0:
                mean_iou = None
                tracked = None
            else:
                mean_iou = sum(ious) / len(ious)
                tracked = count_tracked / len(ious)

            print("mean_iou =", mean_iou)
            print("tracked =", tracked)

            if draw:
                imageio.mimsave(filename, images)
                pygifsicle.optimize(filename)

            return mean_iou, tracked

        for object_id in range(0, min(5, dataset.max_id + 1)):
            mean_iou, tracked = test_object_id(object_id)

            if mean_iou is not None:
                all_mean_ious.append(mean_iou)
                all_tracked.append(tracked)

        if len(all_mean_ious) > 0:
            track_mean_iou = sum(all_mean_ious) / len(all_mean_ious)
            track_mean_tracked = sum(all_tracked) / len(all_tracked)
        else:
            track_mean_iou = None
            track_mean_tracked = None

        print("track_mean_iou =", track_mean_iou)
        print("track_mean_tracked =", track_mean_tracked)

        return track_mean_iou, track_mean_tracked

    tracks = [
        "0000",
        "0001",
        # "0002",
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

    all_ious = []
    all_tracked = []

    for track in tracks:
        track_mean_iou, track_mean_tracked = test_track(track)

        if track_mean_iou is not None:
            all_ious.append(track_mean_iou)
            all_tracked.append(track_mean_tracked)

    total_mean_iou = sum(all_ious) / len(all_ious)
    total_mean_tracked = sum(all_tracked) / len(all_tracked)

    print("total_mean_iou =", total_mean_iou)
    print("total_mean_tracked =", total_mean_tracked)

    print("all_ious =", all_ious)
    print("all_tracked =", all_tracked)


if __name__ == '__main__':

    fire.Fire()
