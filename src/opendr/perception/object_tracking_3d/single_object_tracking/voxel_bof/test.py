import sys
import os
import torch
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.voxel_bof_object_tracking_3d_learner import (
    VoxelBofObjectTracking3DLearner,
)
from opendr.perception.object_detection_3d.datasets.kitti import KittiDataset
from opendr.perception.object_tracking_3d.datasets.kitti_tracking import (
    KittiTrackingDatasetIterator,
    LabeledTrackingPointCloudsDatasetIterator,
)
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.draw import draw_point_cloud_bev
from PIL import Image as PilImage

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

dataset_detection = KittiDataset(dataset_detection_path, subsets_path)
dataset_tracking = LabeledTrackingPointCloudsDatasetIterator(
    dataset_tracking_path + "/training/velodyne/0000",
    dataset_tracking_path + "/training/label_02/0000.txt",
    dataset_tracking_path + "/training/calib/0000.txt",
)
name = "pointpillars_car"
config = all_configs[name]
model_path = model_paths[name]


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

    for i in range(2):
        point_cloud, label = dataset_tracking[i]
        image = draw_point_cloud_bev(point_cloud.data)
        PilImage.fromarray(image).save("./plots/kt_" + str(i) + ".png")


def test_pp_infer_tracking():
    print("Eval", name, "start", file=sys.stderr)

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config, device=DEVICE
    )
    learner.load(model_path)

    res = learner.infer(dataset_tracking[0])

    print(res)


def test_pp_block1():
    print("Eval", name, "start", file=sys.stderr)
    model_path = model_paths[name]

    learner = VoxelBofObjectTracking3DLearner(
        model_config_path=config, device=DEVICE
    )
    learner.load(model_path)


test_draw_tracking_dataset()
