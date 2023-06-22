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


import torch
import yaml
from opendr.perception.skeleton_based_action_recognition.continual_stgcn_learner import (
    CoSTGCNLearner,
    _MODEL_NAMES,
)

from pytorch_benchmark import benchmark
import logging
from typing import List, Union
from opendr.engine.target import Category
from opendr.engine.data import Image

logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_costgcn():
    temp_dir = "./projects/python/perception/skeleton_based_action_recognition/tmp"

    num_runs = 100

    input_shape = (3, 25, 2)  # single skeleton in NTU-RGBD format

    batch_size = 1

    for backbone in _MODEL_NAMES:
        print(f"==== Benchmarking CoSTGCNLearner ({backbone}) ====")

        learner = CoSTGCNLearner(
            device="cuda" if torch.cuda.is_available() else "cpu",
            temp_path=temp_dir,
            backbone=backbone,
            batch_size=batch_size,
        )
        sample = torch.randn(batch_size, *input_shape)  # (B, C, H, W)

        learner.model.eval()
        learner.optimize()

        def get_device_fn(*args):
            nonlocal learner
            return next(learner.model.parameters()).device

        def transfer_to_device_fn(
            sample: Union[torch.Tensor, List[Category], List[Image]],
            device: torch.device,
        ):
            if isinstance(sample, torch.Tensor):
                return sample.to(device=device)

            assert isinstance(sample, list)
            assert isinstance(sample[0], Category)
            return [
                Category(
                    prediction=s.data,
                    confidence=s.confidence.to(device=device),
                )
                for s in sample
            ]

        print("== Benchmarking learner.infer ==")
        results1 = benchmark(
            model=learner.infer,
            sample=sample,
            num_runs=num_runs,
            get_device_fn=get_device_fn,
            transfer_to_device_fn=transfer_to_device_fn,
            print_fn=print,
        )
        print(yaml.dump({"learner.infer": results1}))


if __name__ == "__main__":
    benchmark_costgcn()
