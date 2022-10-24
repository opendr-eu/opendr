import sys
import os
import torch
import fire
from opendr.engine.datasets import PointCloudsDatasetIterator
from opendr.perception.object_detection_3d import VoxelObjectDetection3DLearner
from opendr.perception.object_detection_3d import (
    KittiDataset,
    LabeledPointCloudsDatasetIterator,
)
from projects.python.optimization.channel_pruning import ChannelPruner, prune_learner

config_roots = {
    "pointpillars": os.path.join(
        ".",
        "src",
        "opendr",
        "perception",
        "object_detection_3d",
        "voxel_object_detection_3d",
        "second_detector",
        "configs",
        "pointpillars",
        "car",
    ),
    "tanet": os.path.join(
        ".",
        "src",
        "opendr",
        "perception",
        "object_detection_3d",
        "voxel_object_detection_3d",
        "second_detector",
        "configs",
        "tanet",
        "car",
    ),
}
temp_dir = "./run/models"
subsets_path = os.path.join(
    ".",
    "src",
    "opendr",
    "perception",
    "object_detection_3d",
    "datasets",
    "kitti_subsets",
)
dataset = KittiDataset("/data/sets/kitti_second/", subsets_path)


def train_model(
    model_type,
    device="cuda:0",
    load=0,
    name="pointpillars_car",
    config="preprune_xyres_16.proto",
    samples_list=None,
):

    config = os.path.join(config_roots[model_type], config,)
    model_path = os.path.join(temp_dir, name)

    append_samples_to_name = samples_list is not None

    if samples_list is None:
        samples_list = [1]

    for samples in samples_list:
        if append_samples_to_name:
            model_path = os.path.join(temp_dir, name + "_s" + str(samples))

        learner = VoxelObjectDetection3DLearner(
            model_config_path=config,
            device=device,
            checkpoint_after_iter=1000,
            checkpoint_load_iter=load,
        )

        learner.fit(
            dataset, model_dir=model_path, verbose=True, evaluate=True
        )
        learner.save(model_path)


def train_pointpillars(
    device="cuda:0",
    load=0,
    name="pointpillars_car",
    config="preprune_xyres_16.proto",
    samples_list=None,
):
    return train_model(
        "pointpillars",
        device=device,
        load=load,
        name=name,
        samples_list=samples_list,
        config=config,
    )


def test_model(
    model_type,
    device="cuda:0",
    name="pointpillars_car",
    config="xyres_16.proto",
):

    config = os.path.join(config_roots[model_type], config,)
    model_path = os.path.join(temp_dir, name)

    learner = VoxelObjectDetection3DLearner(
        model_config_path=config, device=device, checkpoint_after_iter=1000
    )
    learner.load(model_path)

    with open(os.path.join(model_path, "eval.txt"), "w") as f:
        result = learner.eval(dataset, verbose=True)
        f.write("3d AP: " + str(result[2][0, :, 0]))
        f.write(str(result))
        f.write("\n\n")


def test_pointpillars(
    device="cuda:0",
    name="pointpillars_car",
    config="xyres_16.proto",
):
    return test_model(
        "pointpillars",
        device=device,
        name=name,
        config=config,
    )


def test_tanet(
    device="cuda:0",
    name="tanet_car",
    config="xyres_16.proto",
):
    return test_model(
        "tanet",
        device=device,
        name=name,
        config=config,
    )


def prune_model(
    model_type,
    device="cuda:0",
    name="pointpillars_car",
    config="prune_xyres_16.proto",
    **kwargs,
):

    config = os.path.join(config_roots[model_type], config,)
    model_path = os.path.join(temp_dir, name)

    learner = VoxelObjectDetection3DLearner(
        model_config_path=config, device=device, checkpoint_after_iter=1000
    )
    learner.load(model_path)

    with open(os.path.join(model_path, "prune.txt"), "w") as f:

        def eval(learner: VoxelObjectDetection3DLearner):
            result = learner.eval(dataset, verbose=True)
            log = "3d AP: " + str(result[2][0, :, 0])
            f.write(log)
            return log, result

        def fine_tune(learner: VoxelObjectDetection3DLearner, _):
            learner.fit(dataset, verbose=True)

        def save(learner: VoxelObjectDetection3DLearner, i):
            learner.save(os.path.join(temp_dir, name, "prune_" + str(i)))

        result = prune_learner(
            learner,
            3,
            eval,
            fine_tune,
            save,
            "local"
        )

        f.write(str(result))
        f.write("\n\n")


def prune_pointpillars(
    device="cuda:0",
    name="pointpillars_car",
    config="prune_xyres_16.proto",
    **kwargs,
):
    return prune_model(
        "pointpillars",
        device=device,
        name=name,
        config=config,
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire()
