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

from opendr.perception.object_detection_2d import Detectron2Learner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", help="Dataset root folder", type=str)
    parser.add_argument("--image-root", help="Dataset root folder", type=str)
    parser.add_argument("--dataset", help="Dataset to train on", type=str)
    parser.add_argument("--backbone", help="Backbone network", type=str, default="resnet", choices=["resnet"])
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=6)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=1e-4)
    parser.add_argument("--val-after", help="Epochs in-between  evaluations", type=int, default=5)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=25)

    args = parser.parse_args()

    detectron2 = Detectron2Learner(device=args.device, batch_size=args.batch_size, lr=args.lr, val_after=args.val_after,
                                   epochs=args.n_epochs, backbone=args.backbone)

    detectron2.fit(args.json_file, args.image_root, args.dataset)
    detectron2.save("./detectron2_saved_model")
