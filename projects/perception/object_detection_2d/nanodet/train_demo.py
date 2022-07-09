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

import argparse

from opendr.engine.datasets import ExternalDataset
from opendr.perception.object_detection_2d import NanodetLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to train on", type=str, default="voc", choices=["voc", "coco",
                                                                                                   "widerperson"])
    parser.add_argument("--data-root", help="Dataset root folder", type=str)
    parser.add_argument("--config", help="Config file root folder", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=6)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=5e-4)
    parser.add_argument("--val-after", help="Epochs in-between  evaluations", type=int, default=5)
    parser.add_argument("--checkpoint-freq", help="Frequency in-between checkpoint saving", type=int, default=5)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=25)
    parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from", type=int, default=0)

    args = parser.parse_args()

    # if args.dataset == 'voc':
    #     dataset = ExternalDataset(args.data_root, 'voc')
    #     val_dataset = ExternalDataset(args.data_root, 'voc')
    # elif args.dataset == 'coco':
    #     dataset = ExternalDataset(args.data_root, 'coco')
    #     val_dataset = ExternalDataset(args.data_root, 'coco')
    # elif args.dataset == 'widerperson':
    #     from opendr.perception.object_detection_2d.datasets import WiderPersonDataset
    #     dataset = WiderPersonDataset(root=args.data_root, splits=['train'])
    #     val_dataset = WiderPersonDataset(root=args.data_root, splits=['val'])

    dataset_root = "/home/manos/data/coco2017"
    dataset = ExternalDataset(dataset_root, 'coco')
    val_dataset = ExternalDataset(dataset_root, 'coco')

    config = "/home/manos/new_opendr/opendr/src/opendr/perception/object_detection_2d/nanodet/algorithm/config/nanodet-plus-m_416.yml"
    nanodet = NanodetLearner(config=config)#, weight_decay=0.05, warmup_steps=500, warmup_ratio=0.0001,
                 #lr_schedule_T_max=300, lr_schedule_eta_min=0.00005, grad_clip=35, iters=300,
                 #batch_size=4, checkpoint_after_iter=50, checkpoint_load_iter=0, temp_path='temp', device='cuda')

    nanodet.fit(dataset, val_dataset)
    nanodet.save()
