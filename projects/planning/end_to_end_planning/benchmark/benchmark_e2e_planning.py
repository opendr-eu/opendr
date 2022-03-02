# Copyright 2020-2022 OpenDR European Project
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
from cv2 import AGAST_FEATURE_DETECTOR_AGAST_5_8
import yaml
import torch
import logging
from pytorch_benchmark import benchmark
from opendr.planning.end_to_end_planning.e2e_planning_learner import EndToEndPlanningRLLearner
from opendr.planning.end_to_end_planning.envs.agi_env import AgiEnv


logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_e2e_planner():
    root_dir = "./projects/planning/end_to_end_planning/benchmark"
    media_dir = root_dir + "/media"
    num_runs = 100
    env = AgiEnv()

    models = [
        "e2e_planner",
    ]

    batch_size = 1

    sample = env.observation_space.sample()

    if os.path.exists(root_dir + "/results_ab3dmot.txt"):
        os.remove(root_dir + "/results_ab3dmot.txt")

    for model_name in models:
        print(f"==== Benchmarking EndToEndPlanningRLLearner ({model_name}) ====")

        learner = EndToEndPlanningRLLearner(env)

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
            sample=sample,
            sample_with_batch_size1=sample,
            num_runs=num_runs,
            get_device_fn=get_device_fn,
            transfer_to_device_fn=transfer_to_device_fn,
            batch_size=batch_size,
        )

        print(yaml.dump({"learner.infer": results1}))

        with open(root_dir + "/results_e2e_planner.txt", "a") as f:
            print(f"==== Benchmarking EndToEndPlanningRLLearner ({model_name}) ====", file=f)
            print(yaml.dump({"learner.infer": results1}), file=f)
            print("\n\n", file=f)

    print("===END===")


if __name__ == "__main__":
    benchmark_e2e_planner()
