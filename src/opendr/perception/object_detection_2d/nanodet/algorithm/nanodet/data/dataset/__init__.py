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
from opendr.engine.datasets import ExternalDataset

from opendr.perception.object_detection_2d.datasets.detection_dataset import DetectionDataset
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.dataset.coco import CocoDataset
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.dataset.xml_dataset import XMLDataset


def build_dataset(cfg, dataset, class_names, mode, verbose=True):
        dataset_cfg = copy.deepcopy(cfg)
        supported_datasets = ['coco', 'voc']
        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() not in supported_datasets:
                raise UserWarning("ExternalDataset dataset_type must be one of: ", supported_datasets)

            dataset_root = dataset.path
            img_path = dataset.path + dataset_cfg.pop("img_path")
            ann_path = dataset.path + dataset_cfg.pop("ann_path")

            if verbose:
                print("Loading {} type dataset...".format(dataset.dataset_type))
                print("From {}".format(dataset_root))

            if dataset.dataset_type.lower() == 'voc':
                dataset = XMLDataset(img_path=img_path, ann_path=ann_path, mode=mode,
                                     class_names=class_names ,**dataset_cfg)

            elif dataset.dataset_type.lower() == 'coco':
                dataset = CocoDataset(img_path=img_path, ann_path=ann_path, mode=mode, **dataset_cfg)
            if verbose:
                print("ExternalDataset loaded.")
            return dataset
        else:
            raise ValueError("Dataset type {} not supported".format(type(dataset)))