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
from opendr.perception.object_tracking_2d import ObjectTracking2DFairMotLearner
from opendr.engine.data import Image

logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_fair_mot():
    root_dir = "./projects/python/perception/object_tracking_2d/benchmark"
    temp_dir = root_dir + "/tmp"
    models_dir = root_dir + "/models"
    media_dir = root_dir + "/media"
    num_runs = 100

    models = [
        "crowdhuman_dla34",
        "fairmot_dla34",
    ]

    batch_size = 2

    sample = Image.open(media_dir + "/000001.jpg")
    samples = [sample for _ in range(batch_size)]

    if os.path.exists(root_dir + "/results_fair_mot.txt"):
        os.remove(root_dir + "/results_fair_mot.txt")

    for model_name in models:
        print(f"==== Benchmarking ObjectTracking2DFairMotLearner ({model_name}) ====")

        learner = ObjectTracking2DFairMotLearner(
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

        inner_fps = learner.infers_count / (learner.infers_time)

        print("Inner FPS =", inner_fps)
        print(yaml.dump({"learner.infer": results1}))

        with open(root_dir + "/results_fair_mot.txt", "a") as f:
            print(f"==== Benchmarking ObjectTracking2DFairMotLearner ({model_name}) ====", file=f)
            print("Inner FPS =", inner_fps, file=f)
            print(yaml.dump({"learner.infer": results1}), file=f)
            print("\n\n", file=f)

        # print("== Benchmarking model directly ==")
        # results2 = benchmark(learner.model, sample, num_runs=num_runs)
        # print(yaml.dump({"learner.model.forward": results2}))

    print("===END===")


if __name__ == "__main__":
    benchmark_fair_mot()
