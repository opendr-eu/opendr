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
from opendr.perception.object_detection_2d import DetrLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", help="Backbone network", type=str, default="resnet50",
                        choices=["resnet50", "resnet101"])
    parser.add_argument("--data-root", help="Dataset root folder", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=6)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=1e-4)

    args = parser.parse_args()

    dataset = ExternalDataset(args.data_root, 'coco')
    val_dataset = ExternalDataset(args.data_root, 'coco')

    learner = DetrLearner(device=args.device, batch_size=args.batch_size, lr=args.lr, backbone=args.backbone)

    learner.fit(dataset, val_dataset)
    learner.save("saved_detr_model")
