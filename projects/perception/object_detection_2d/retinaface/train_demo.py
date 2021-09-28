# Copyright 2020-2021 OpenDR European Project
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
import argparse

from opendr.perception.object_detection_2d.retinaface.retinaface_learner import RetinaFaceLearner
from opendr.perception.object_detection_2d.datasets import WiderFaceDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Dataset root folder, only WIDERFace dataset is supported", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=6)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=5e-4)
    parser.add_argument("--val-after", help="Epochs in-between  evaluations", type=int, default=5)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=1)

    args = parser.parse_args()

    dataset = WiderFaceDataset(root=args.data_root, splits=['train'])

    face_learner = RetinaFaceLearner(backbone='resnet', prefix='retinaface_resnet50',
                                     epochs=args.n_epochs, log_after=10, flip=False, shuffle=False,
                                     lr=args.lr, lr_steps='55,68,80', weight_decay=5e-4,
                                     batch_size=4, val_after=args.val_after,
                                     temp_path='temp_retinaface', checkpoint_after_iter=1)

    face_learner.fit(dataset, val_dataset=dataset, verbose=True)
    face_learner.save(os.path.join("pretrained", "retinaface_resnet50"))
