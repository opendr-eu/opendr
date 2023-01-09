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
import torch
import yaml
from pytorch_benchmark import benchmark
import logging
import argparse

# opendr imports
from opendr.perception.facial_expression_recognition import FacialEmotionLearner
from opendr.engine.data import Image


logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_esr(args):
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    device = args.device

    print(f"==== Benchmarking {args.method} ====")

    learner = FacialEmotionLearner(device=device, ensemble_size=args.ensemble_size, diversify=True)
    learner.init_model(num_branches=args.ensemble_size)

    if device == 'cuda':
        learner.model.cuda()

    num_runs = 100
    batch_size = 32
    C = 3
    H = 96
    W = 96
    input_face = torch.randn(C, H, W)
    input_img = Image(input_face)
    input_batch = []
    for i in range(batch_size):
        input_batch.append(input_img)
    if type(input_batch) is list:
        input_batch = torch.stack([torch.tensor(v.data) for v in input_batch])

    def get_device_fn(*args):
        # nonlocal learner
        return next(learner.model.parameters()).device

    def transfer_to_device_fn(sample, device,):
        return sample

    print("== Benchmarking learner.infer ==")
    results1 = benchmark(model=learner.infer,
                         sample=input_batch,
                         sample_with_batch_size1=None,
                         num_runs=num_runs,
                         get_device_fn=get_device_fn,
                         transfer_to_device_fn=transfer_to_device_fn,
                         batch_size=batch_size,
                         print_fn=print,
                         )
    with open(results_dir + f"/benchmark_{args.method}_{device}.txt", "a") as f:
        print("== Benchmarking learner.infer ==", file=f)
        print(yaml.dump({"learner.infer": results1}), file=f)
        print("\n\n", file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument('--method', type=str, default='div_esr_9',
                        help='action detection method')
    parser.add_argument('--ensemble_size', type=int, default=9,
                        help='number of ensemble branches')

    args = parser.parse_args()
    benchmark_esr(args)
