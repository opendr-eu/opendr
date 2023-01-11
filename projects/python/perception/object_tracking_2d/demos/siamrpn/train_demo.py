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

import argparse

from opendr.engine.datasets import ExternalDataset
from opendr.perception.object_tracking_2d import SiamRPNLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", help="Dataset to train on. To train with multiple, separate with ','",
                        type=str, default="coco")
    parser.add_argument("--data-root", help="Dataset root folder", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=6)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=5e-4)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=25)

    args = parser.parse_args()

    if ',' in args.datasets:
        dataset = [ExternalDataset(args.data_root, dataset_type) for dataset_type in args.datasets.split(',')]
    else:
        dataset = ExternalDataset(args.data_root, args.datasets)

    learner = SiamRPNLearner(device=args.device, n_epochs=args.n_epochs, batch_size=args.batch_size,
                             lr=args.lr)

    learner.fit(dataset)
    learner.save("siamrpn_custom")
