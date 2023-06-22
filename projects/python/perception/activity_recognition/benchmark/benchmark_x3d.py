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
from opendr.perception.activity_recognition import X3DLearner

from pytorch_benchmark import benchmark
import logging
from typing import List, Union
from opendr.engine.target import Category
from opendr.engine.data import Video

logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_x3d():
    temp_dir = "./projects/python/perception/activity_recognition/benchmark/tmp"

    num_runs = 100

    # As found in src/opendr/perception/activity_recognition/x3d/hparams
    input_shape = {
        "xs": (3, 4, 160, 160),
        "s": (3, 13, 160, 160),
        "m": (3, 16, 224, 224),
        "l": (3, 16, 312, 312),
    }

    # Max power of 2
    # batch_size = {  # RTX2080Ti
    #     "xs": 32,
    #     "s": 16,
    #     "m": 8,
    #     "l": 2,
    # }
    # batch_size = {  # TX2
    #     "xs": 16,
    #     "s": 8,
    #     "m": 4,
    #     "l": 2,
    # }
    # batch_size = {  # Xavier
    #     "xs": 32,
    #     "s": 16,
    #     "m": 8,
    #     "l": 2,
    # }
    batch_size = {  # CPU - larger batch sizes don't increase throughput
        "xs": 1,
        "s": 1,
        "m": 1,
        "l": 1,
    }

    for backbone in ["xs", "s", "m", "l"]:
        print(f"==== Benchmarking X3DLearner ({backbone}) ====")

        learner = X3DLearner(
            device="cuda" if torch.cuda.is_available() else "cpu",
            temp_path=temp_dir,
            backbone=backbone,
            batch_size=batch_size[backbone],
        )
        learner.optimize()
        learner.model.eval()

        sample = torch.randn(
            batch_size[backbone], *input_shape[backbone]
        )  # (B, C, T, H, W)
        # video_samples = [Video(v) for v in sample]
        # video_sample = [Video(sample[0])]

        def get_device_fn(*args):
            nonlocal learner
            return next(learner.model.parameters()).device

        def transfer_to_device_fn(
            sample: Union[torch.Tensor, List[Category], List[Video]],
            device: torch.device,
        ):
            if isinstance(sample, torch.Tensor):
                return sample.to(device=device)

            assert isinstance(sample, list)

            if isinstance(sample[0], Video):
                # Video.data i a numpy array, which is always on CPU
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
            # sample_with_batch_size1=sample[0].unsqueeze(0),
            num_runs=num_runs,
            get_device_fn=get_device_fn,
            transfer_to_device_fn=transfer_to_device_fn,
            batch_size=batch_size[backbone],
            print_fn=print,
        )
        print(yaml.dump({"learner.infer": results1}))


if __name__ == "__main__":
    benchmark_x3d()
