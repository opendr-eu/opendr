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
from opendr.perception.activity_recognition import CoTransEncLearner

from pytorch_benchmark import benchmark
import logging
from typing import List, Union
from opendr.engine.target import Category
from opendr.engine.data import Image

logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_cotransenc():
    temp_dir = "./projects/python/perception/activity_recognition/benchmark/tmp"
    num_runs = 100
    batch_size = 1

    for num_layers in [1, 2]:  # --------- A few plausible hparams ----------
        for (input_dims, sequence_len) in [(1024, 32), (2048, 64), (4096, 64)]:
            print(
                f"==== Benchmarking CoTransEncLearner (l{num_layers}-d{input_dims}-t{sequence_len}) ===="
            )
            learner = CoTransEncLearner(
                device="cuda" if torch.cuda.is_available() else "cpu",
                temp_path=temp_dir + f"/{num_layers}_{input_dims}_{sequence_len}",
                num_layers=num_layers,
                input_dims=input_dims,
                hidden_dims=input_dims // 2,
                sequence_len=sequence_len,
                num_heads=input_dims // 128,
                batch_size=batch_size,
            )
            learner.optimize()

            sample = torch.randn(1, input_dims)

            # Warm-up continual inference not needed for optimized version:
            # for _ in range(sequence_len - 1):
            #     learner.infer(sample)

            def get_device_fn(*args):
                nonlocal learner
                return next(learner.model.parameters()).device

            def transfer_to_device_fn(
                sample: Union[torch.Tensor, List[Category], List[Image]],
                device: torch.device,
            ):
                if isinstance(sample, torch.Tensor):
                    return sample.to(device=device)

                assert isinstance(sample, Category)
                return Category(
                    prediction=sample.data,
                    confidence=sample.confidence.to(device=device),
                )

            results1 = benchmark(
                model=learner.infer,
                sample=sample,
                num_runs=num_runs,
                get_device_fn=get_device_fn,
                transfer_to_device_fn=transfer_to_device_fn,
                batch_size=batch_size,
                print_fn=print,
            )
            print(yaml.dump({"learner.infer": results1}))


if __name__ == "__main__":
    benchmark_cotransenc()
