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

import argparse

from opendr.perception.object_detection_2d.retinaface.retinaface_learner import RetinaFaceLearner
from opendr.perception.object_detection_2d.datasets import WiderFaceDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Dataset root folder, only WIDERFace dataset is supported", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backbone", help="Network backbone", type=str, default="resnet", choices=["resnet", "mnet"])
    parser.add_argument("--pyramid", help="Image pyramid flag", dest='pyramid', action='store_true', default=False)
    parser.add_argument("--flip", help="Image flip flag", dest='flip', action='store_true', default=False)

    args = parser.parse_args()

    dataset = WiderFaceDataset(root=args.data_root, splits=['val'])

    learner = RetinaFaceLearner(backbone=args.backbone, device=args.device)
    learner.download(".", mode="pretrained")
    learner.load("./retinaface_{}".format(args.backbone))

    eval_results = learner.eval(dataset, use_subset=True, subset_size=100,
                                flip=args.flip, pyramid=args.pyramid,
                                verbose=True)
    for k, v in eval_results.items():
        print(k, v)
