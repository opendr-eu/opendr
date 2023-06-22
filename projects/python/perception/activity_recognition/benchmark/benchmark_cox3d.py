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
from opendr.perception.activity_recognition import CoX3DLearner

from pytorch_benchmark import benchmark
import logging
from typing import List, Union
from opendr.engine.target import Category
from opendr.engine.data import Image

logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_cox3d():
    temp_dir = "./projects/python/perception/activity_recognition/benchmark/tmp"

    num_runs = 100

    # As found in src/opendr/perception/activity_recognition/x3d/hparams
    input_shape = {
        "xs": (3, 160, 160),
        "s": (3, 160, 160),
        "m": (3, 224, 224),
        "l": (3, 312, 312),
    }

    # Max power of 2
    # batch_size = {  # RTX2080Ti
    #     "xs": 128,
    #     "s": 64,
    #     "m": 64,
    #     "l": 8,
    # }
    # batch_size = {  # TX2
    #     "xs": 32,
    #     "s": 16,
    #     "m": 8,
    #     "l": 4,
    # }
    # batch_size = {  # Xavier
    #     "xs": 64,
    #     "s": 32,
    #     "m": 16,
    #     "l": 8,
    # }
    batch_size = {  # CPU - larger batch sizes don't increase throughput
        "xs": 1,
        "s": 1,
        "m": 1,
        "l": 1,
    }

    for backbone in ["s", "m", "l"]:
        print(f"==== Benchmarking CoX3DLearner ({backbone}) ====")

        learner = CoX3DLearner(
            device="cuda" if torch.cuda.is_available() else "cpu",
            temp_path=temp_dir,
            backbone=backbone,
        )
        learner.optimize()

        sample = torch.randn(
            batch_size[backbone], *input_shape[backbone]
        )  # (B, C, H, W)
        # image_samples = [Image(v) for v in sample]
        # image_sample = [Image(sample[0])]

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

            if isinstance(sample[0], Image):
                # Image.data i a numpy array, which is always on CPU
                return sample

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
            # sample_with_batch_size1=image_sample,
            num_runs=num_runs,
            get_device_fn=get_device_fn,
            transfer_to_device_fn=transfer_to_device_fn,
            batch_size=batch_size[backbone],
            print_fn=print,
        )
        print(yaml.dump({"learner.infer": results1}))


if __name__ == "__main__":
    benchmark_cox3d()
