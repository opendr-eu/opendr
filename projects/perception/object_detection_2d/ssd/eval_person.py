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

from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner
from opendr.perception.object_detection_2d.yolov3.yolov3_learner import YOLOv3DetectorLearner
from opendr.perception.object_detection_2d.centernet.centernet_learner import CenterNetDetectorLearner
from opendr.perception.object_detection_2d.datasets import WiderPersonDataset
from opendr.perception.object_detection_2d.datasets.xmldataset import XMLBasedDataset
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
from opendr.perception.object_detection_2d.utils.eval_utils import MeanAveragePrecision
from opendr.perception.object_detection_2d.datasets.detection_dataset import ConcatDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Dataset root folder", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    # val_dataset = WiderPersonDataset(root='/home/administrator/data/wider_person', splits=['val'])
    val_dataset = XMLBasedDataset(root='/home/administrator/data/agi_human_data', dataset_type='agi_human',
                                           images_dir='human', annotations_dir='human_anot', classes=['person'])
    val_dataset_no_human = XMLBasedDataset(root='/home/administrator/data/agi_human_data', dataset_type='agi_human',
                                  images_dir='no_human', annotations_dir='no_human_anot', classes=['person'])
    val_dataset = ConcatDataset([val_dataset, val_dataset_no_human])
    print(val_dataset.classes)
    # metric = VOCMApMetric(class_names=val_dataset.classes, iou_thresh=0.45)
    metric = None

    detector = SingleShotDetectorLearner(device=args.device, backbone='mobilenet1.0')
    # detector = YOLOv3DetectorLearner(device=args.device)
    # detector = CenterNetDetectorLearner(device=args.device)
    # detector.download(".", mode="pretrained")
    # detector.load("./centernet_default", verbose=True)
    detector.load_gcv('coco', val_dataset.classes)
    # detector.save("yolov3_voc_person")
    results = detector.eval(val_dataset)
    for k, v in results.items():
        print('{}: {}'.format(k, v))
