from engine.datasets import ExternalDataset
from perception.object_detection_3d.datasets.kitti import KittiDataset
from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner,
)
import torch


def test_training():
    dataset_path = "/data/sets/opendr_kitti"
    tanet_path = "./perception/object_detection_3d/voxel_object_detection_3d/models/learning_tanet_16_car_1"
    tanet_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/tanet/car/xyres_16.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(model_config_path=tanet_config_path)
    learner.fit(dataset,)


def test_training_short():
    dataset_path = "/data/sets/opendr_kitti"
    tanet_path = "./perception/object_detection_3d/voxel_object_detection_3d/models/short_learning_tanet_16_car_4"
    tanet_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/tanet/car/test_short.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(model_config_path=tanet_config_path)
    # learner.load(tanet_path)
    starting_param = list(learner.model.parameters())[0].clone()
    learner.fit(dataset, auto_save=True, model_dir=tanet_path, verbose=True)
    new_param = list(learner.model.parameters())[0].clone()
    print(not torch.equal(starting_param, new_param))
    # learner.save(tanet_path)
    pass


def test_training_pointpillars():
    dataset_path = "/data/sets/opendr_kitti"
    pp_path = "./perception/object_detection_3d/voxel_object_detection_3d/models/short_learning_pointpillars_16_car"
    pp_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/pointpillars/car/xyres_16.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(model_config_path=pp_config_path)
    learner.load(pp_path)
    learner.fit(dataset)


def test_training_pointpillars_short():
    dataset_path = "/data/sets/opendr_kitti"
    pp_path = "./perception/object_detection_3d/voxel_object_detection_3d/models/learning_pointpillars_16_car"
    pp_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/pointpillars/car/test_short.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(model_config_path=pp_config_path)
    learner.load(pp_path)
    learner.fit(dataset)


def test_eval():
    dataset_path = "/data/sets/opendr_kitti"
    tanet_path = (
        "./perception/object_detection_3d/voxel_object_detection_3d/models/tanet_16_car"
    )
    # tanet_path = "/home/io/Detection/Models/tanet/car_tr_1/car_trained_model"
    tanet_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/tanet/car/xyres_16.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(model_config_path=tanet_config_path)
    learner.load(tanet_path, logging_path=tanet_path + "/lololog.txt")
    mAPbbox, mAPbev, mAP3d, mAPaos = learner.eval(dataset)

    print(mAP3d[0][0][0] > 85 and mAP3d[0][0][0] < 90)
    pass


def test_infer():

    from perception.object_detection_3d.voxel_object_detection_3d.second.run import (
        example_convert_to_torch,
    )
    from perception.object_detection_3d.voxel_object_detection_3d.second.data.preprocess import (
        merge_second_batch,
    )

    dataset_path = "/data/sets/opendr_kitti"
    tanet_path = (
        "./perception/object_detection_3d/voxel_object_detection_3d/models/tanet_16_car"
    )
    # tanet_path = "/home/io/Detection/Models/tanet/car_tr_1/car_trained_model"
    tanet_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/tanet/car/xyres_16.proto"
    dataset = KittiDataset(dataset_path)

    learner = VoxelObjectDetection3DLearner(model_config_path=tanet_config_path)
    learner.load(tanet_path)

    (_, eval_dataset_iterator, ground_truth_annotations,) = learner._prepare_datasets(
        None,
        dataset,
        learner.input_config,
        learner.evaluation_input_config,
        learner.model_config,
        learner.voxel_generator,
        learner.target_assigner,
        None,
        require_dataset=False,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset_iterator,
        batch_size=learner.evaluation_input_config.batch_size,
        shuffle=False,
        num_workers=learner.evaluation_input_config.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
    )

    result = learner.infer(
        example_convert_to_torch(next(iter(eval_dataloader)), learner.float_dtype)
    )

    print(len(result[1][0]["bbox"]) > 0)

    pass


# test_infer()
test_training_short()
