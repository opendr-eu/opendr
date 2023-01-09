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

# opendr imports
import argparse
from opendr.perception.skeleton_based_action_recognition import ProgressiveSpatioTemporalGCNLearner
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner

logger = logging.getLogger("benchmark")
logging.basicConfig()
logger.setLevel("DEBUG")


def benchmark_stgcn(args):
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    device = args.device
    if args.method == 'pstgcn':
        print(f"==== Benchmarking pstgcn ({args.dataset_name}) ====")
        if args.dataset_name == 'nturgbd_cv':
            learner = ProgressiveSpatioTemporalGCNLearner(device=args.device, dataset_name='nturgbd_cv',
                                                          topology=[5, 4, 5, 2, 3, 4, 3, 4], in_channels=3,
                                                          num_point=25, graph_type='ntu')
        elif args.dataset_name == 'nturgbd_cs':
            learner = ProgressiveSpatioTemporalGCNLearner(device=args.device, dataset_name='nturgbd_cs',
                                                          topology=[5, 4, 3, 5, 3, 5, 7, 4], in_channels=3,
                                                          num_point=25, graph_type='ntu')
    else:
        print(f"==== Benchmarking {args.method} ({args.dataset_name}) ====")
        learner = SpatioTemporalGCNLearner(device=args.device, dataset_name='nturgbd_cv',
                                           method_name=args.method, in_channels=3, num_point=25,
                                           graph_type='ntu', num_class=60, num_subframes=100)

    learner.init_model()
    if args.device == 'cuda':
        learner.model.cuda()

    batch_size = 1
    num_runs = 100
    C = 3
    T = 300
    V = 25
    M = 2
    data = torch.randn(batch_size, C, T, V, M)
    samples = data

    def get_device_fn(*args):
        nonlocal learner
        return next(learner.model.parameters()).device

    def transfer_to_device_fn(sample, device,):
        return sample

    print("== Benchmarking learner.infer ==")
    results1 = benchmark(model=learner.infer,
                         sample=samples,
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
        print("== Benchmarking model directly ==", file=f)
        results2 = benchmark(learner.model, data, num_runs=num_runs, print_fn=print)
        print(yaml.dump({"learner.model.forward": results2}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument('--method', type=str, default='stgcn',
                        help='action detection method')
    parser.add_argument('--dataset_name', type=str, default='nturgbd_cv',
                        help='action detection method')

    args = parser.parse_args()
    benchmark_stgcn(args)
