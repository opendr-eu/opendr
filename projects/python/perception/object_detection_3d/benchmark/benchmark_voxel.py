# Copyright 2020-2023 OpenDR European Project
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

import os
import yaml
import torch
import logging
from pytorch_benchmark import benchmark
from opendr.perception.object_detection_3d import VoxelObjectDetection3DLearner
from opendr.engine.datasets import PointCloudsDatasetIterator

logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_voxel():
    root_dir = "./projects/python/perception/object_detection_3d/benchmark"
    temp_dir = root_dir + "/tmp"
    configs_dir = root_dir + "/configs"
    models_dir = root_dir + "/models"
    media_dir = root_dir + "/media"
    num_runs = 100

    models = [
        ["pointpillars_car_xyres_16", "pointpillars_car_xyres_16.proto"],
        ["pointpillars_ped_cycle_xyres_16", "pointpillars_ped_cycle_xyres_16.proto"],
        ["tanet_car_xyres_16", "tanet_car_xyres_16.proto"],
        ["tanet_car_xyres_16", "tanet_car_xyres_16_near_0.24.proto"],
        ["tanet_ped_cycle_xyres_16", "tanet_ped_cycle_xyres_16.proto"],
    ]

    batch_size = 2

    dataset = PointCloudsDatasetIterator(media_dir)
    sample = dataset[0]
    samples = [dataset[0] for _ in range(batch_size)]

    if os.path.exists(root_dir + "/results_voxel.txt"):
        os.remove(root_dir + "/results_voxel.txt")

    for model_name, config in models:
        print(f"==== Benchmarking VoxelObjectDetection3DLearner ({config}) ====")

        config_path = configs_dir + "/" + config

        learner = VoxelObjectDetection3DLearner(
            config_path,
            temp_path=temp_dir,
        )

        if model_name is not None and not os.path.exists(
            models_dir + "/" + model_name
        ):
            learner.download(model_name, models_dir)
        learner.load(models_dir + "/" + model_name, verbose=True)

        def get_device_fn(*args):
            nonlocal learner
            return torch.device(learner.device)

        def transfer_to_device_fn(
            sample,
            device,
        ):
            return sample

        print("== Benchmarking learner.infer ==")
        results1 = benchmark(
            model=learner.infer,
            sample=samples,
            sample_with_batch_size1=sample,
            num_runs=num_runs,
            get_device_fn=get_device_fn,
            transfer_to_device_fn=transfer_to_device_fn,
            batch_size=batch_size,
        )

        inner_fps = (
            learner.model._total_inference_count /
            (learner.model._total_forward_time + learner.model._total_postprocess_time)
        )

        print("Inner FPS =", inner_fps)
        print(yaml.dump({"learner.infer": results1}))

        with open(root_dir + "/results_voxel.txt", "a") as f:
            print(f"==== Benchmarking VoxelObjectDetection3DLearner ({config}) ====", file=f)
            print("Inner FPS =", inner_fps, file=f)
            print(yaml.dump({"learner.infer": results1}), file=f)
            print("\n\n", file=f)

        # print("== Benchmarking model directly ==")
        # results2 = benchmark(learner.model, sample, num_runs=num_runs)
        # print(yaml.dump({"learner.model.forward": results2}))

    print("===END===")


if __name__ == "__main__":
    benchmark_voxel()
