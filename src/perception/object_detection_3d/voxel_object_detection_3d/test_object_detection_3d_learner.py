from engine.datasets import ExternalDataset
from perception.object_detection_3d.datasets.kitti import KittiDataset
from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner,
)


def test_training():
    dataset_path = "/data/sets/opendr_kitti"
    tanet_path = "./perception/object_detection_3d/voxel_object_detection_3d/models/learning_tanet_16_car"
    tanet_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/tanet/car/xyres_16.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(
        model_config_path=tanet_config_path
    )
    learner.load(tanet_path)
    learner.fit(dataset)


def test_training_short():
    dataset_path = "/data/sets/opendr_kitti"
    tanet_path = "./perception/object_detection_3d/voxel_object_detection_3d/models/short_learning_tanet_16_car"
    tanet_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/tanet/car/xyres_16.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(
        model_config_path=tanet_config_path
    )
    learner.load(tanet_path)
    learner.fit(dataset)


def test_eval():
    dataset_path = "/data/sets/opendr_kitti"
    tanet_path = "./perception/object_detection_3d/voxel_object_detection_3d/models/tanet_16_car"
    # tanet_path = "/home/io/Detection/Models/tanet/car_tr_1/car_trained_model"
    tanet_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/tanet/car/xyres_16.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(
        model_config_path=tanet_config_path
    )
    learner.load(tanet_path)
    learner.eval(dataset)


test_training()

