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
from opendr.perception.object_detection_2d import NanodetLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to train on", type=str, default="coco", choices=["voc", "coco"])
    parser.add_argument("--data-root", help="Dataset root folder", type=str)
    parser.add_argument("--model", help="Model for which a config file will be used", type=str, default="m")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=6)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=5e-4)
    parser.add_argument("--checkpoint-freq", help="Frequency in-between checkpoint saving and evaluations",
                        type=int, default=50)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=300)
    parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from",
                        type=int, default=0)

    args = parser.parse_args()

    dataset = ExternalDataset(args.data_root, args.dataset)
    val_dataset = ExternalDataset(args.data_root, args.dataset)

    nanodet = NanodetLearner(model_to_use=args.model, iters=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
                             checkpoint_after_iter=args.checkpoint_freq, checkpoint_load_iter=args.resume_from,
                             device=args.device)

    nanodet.download("./predefined_examples", mode="pretrained")
    nanodet.load("./predefined_examples/nanodet_{}".format(args.model), verbose=True)
    nanodet.fit(dataset, val_dataset)
    nanodet.save()
