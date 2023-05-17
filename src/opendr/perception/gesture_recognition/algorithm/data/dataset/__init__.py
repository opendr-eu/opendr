# Modifications Copyright 2023 - present, OpenDR European Project
#
# Copyright 2021 RangiLyu.
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


import copy
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.dataset.coco import CocoDataset


def build_dataset(cfg, dataset, class_names, mode, verbose=True, preprocess=True, download=False):
    dataset_cfg = copy.deepcopy(cfg)
    if verbose:
        print("Loading type dataset from {}".format(dataset.path))

    if mode == "train":
        img_path = "{}/train".format(dataset.path)
        ann_path = "{}/train.json".format(dataset.path)
    elif mode == "val":
        img_path = "{}/val".format(dataset.path)
        ann_path = "{}/val.json".format(dataset.path)
    else:
        img_path = "{}/test".format(dataset.path)
        ann_path = "{}/test.json".format(dataset.path)
    dataset = CocoDataset(img_path=img_path, ann_path=ann_path, mode=mode, **dataset_cfg)

    if verbose:
        print("ExternalDataset loaded.")
    return dataset
