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

from opendr.perception.object_detection_2d import NanodetLearner
from opendr.perception.object_detection_2d import WiderPersonDataset
from opendr.engine.datasets import ExternalDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Dataset root folder", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    # dataset_root = "/home/manos/data/coco2017"
    # val_dataset = ExternalDataset(dataset_root, 'coco')

    dataset_root = "/home/manos/data/cocoDataset/temp"
    val_dataset = ExternalDataset(dataset_root, 'voc')

    config = "/home/manos/new_opendr/opendr/src/opendr/perception/object_detection_2d/nanodet/algorithm/config/nanodet-plus-m_416.yml"
    nanodet = NanodetLearner(config=config)  # , weight_decay=0.05, warmup_steps=500, warmup_ratio=0.0001,
    # lr_schedule_T_max=300, lr_schedule_eta_min=0.00005, grad_clip=35, iters=300,
    # batch_size=4, checkpoint_after_iter=50, checkpoint_load_iter=0, temp_path='temp', device='cuda')

    # nanodet.download(".", mode="pretrained")
    nanodet.load("/home/manos/new_opendr/opendr/src/opendr/perception/object_detection_2d/nanodet/algorithm/pretrained_models/nanodet-plus-m_416_checkpoint.ckpt", verbose=True)
    nanodet.eval(val_dataset)
