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
dataset_path = "/data/sets/kitti_second/"
dataset = KittiDataset(dataset_path, subsets_path)


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

        learner.fit(dataset, model_dir=model_path, verbose=True, evaluate=True)
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
    model_type, device="cuda:0", name="pointpillars_car", config="xyres_16.proto",
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
        f.write("\n")
        f.write(str(result))
        f.write("\n\n")


def test_pointpillars(
    device="cuda:0", name="pointpillars_car", config="preprune_xyres_16.proto",
):
    return test_model("pointpillars", device=device, name=name, config=config,)


def test_tanet(
    device="cuda:0", name="tanet_car", config="xyres_16.proto",
):
    return test_model("tanet", device=device, name=name, config=config,)


def visualize_model(
    model_type, device="cuda:0", name="pointpillars_car", config="xyres_16.proto",
):

    # import hiddenlayer as hl
    from torchviz import make_dot
    from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.data.preprocess import (
        merge_second_batch,
    )
    from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.run import (
        example_convert_to_torch,
    )

    config = os.path.join(config_roots[model_type], config,)
    model_path = os.path.join(temp_dir, name)

    dataset = LabeledPointCloudsDatasetIterator(
        dataset_path + "/training/velodyne_reduced",
        dataset_path + "/training/label_2",
        dataset_path + "/training/calib",
    )

    learner = VoxelObjectDetection3DLearner(
        model_config_path=config, device=device, checkpoint_after_iter=1000
    )
    learner.load(model_path)

    point_cloud, _ = dataset[0]
    learner.infer(point_cloud)

    input_data = merge_second_batch(
        [learner.infer_point_cloud_mapper(point_cloud.data)]
    )
    input_data = example_convert_to_torch(
        input_data, learner.float_dtype, device=learner.device,
    )

    output = learner.model(input_data)
    os.makedirs("./run/plots/graphs", exist_ok=True)

    make_dot(output[0]["box3d_lidar"], params=dict(list(learner.model.named_parameters()))).render(
        "./run/plots/graphs/" + name, format="png"
    )
    # graph = hl.build_graph(model=learner.model, args=input_data)
    # graph.theme = hl.graph.THEMES["blue"].copy()
    # graph.save("./run/plots/graphs/" + name, format="png")


def visualize_pointpillars(
    device="cuda:0", name="pointpillars_car", config="preprune_xyres_16.proto",
):
    return visualize_model("pointpillars", device=device, name=name, config=config,)


def visualize_tanet(
    device="cuda:0", name="tanet_car", config="xyres_16.proto",
):
    return visualize_model("tanet", device=device, name=name, config=config,)


def prune_model(
    model_type,
    device="cuda:0",
    name="pointpillars_car",
    config="prune_xyres_16.proto",
    steps=5,
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

        def fine_tune(learner: VoxelObjectDetection3DLearner, i):
            model_dir = os.path.join(model_path, "prune_" + str(i))
            learner.fit(dataset, verbose=True, model_dir=model_dir)

        def save(learner: VoxelObjectDetection3DLearner, i):
            model_dir = os.path.join(model_path, "prune_" + str(i))
            learner.save(model_dir)

        result = prune_learner(learner, steps, eval, fine_tune, save, "local", **kwargs)

        f.write(str(result))
        f.write("\n\n")


def prune_pointpillars(
    device="cuda:0", name="pointpillars_car", config="prune_xyres_16.proto", **kwargs,
):
    return prune_model(
        "pointpillars", device=device, name=name, config=config, **kwargs,
    )


if __name__ == "__main__":
    fire.Fire()
