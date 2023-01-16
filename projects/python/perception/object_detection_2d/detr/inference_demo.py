#!/usr/bin/env python
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

import numpy as np
import urllib
import cv2
from opendr.perception.object_detection_2d import DetrLearner
from opendr.perception.object_detection_2d.detr.algorithm.util.draw import draw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backbone", help="Backbone to use (resnet50, resnet101)", type=str, default="resnet101",
                        choices=["resnet50", "resnet101"])
    parser.add_argument("--panoptic-segmentation", dest='panoptic_segmentation', help="Perform panoptic segmentation",
                        default=False, action='store_true')

    args = parser.parse_args()

    # Download an image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)

    # For panoptic segmentation, the number of classes is different
    if args.panoptic_segmentation:
        num_classes = 250
    else:
        num_classes = 91

    learner = DetrLearner(
        backbone=args.backbone,
        device=args.device,
        panoptic_segmentation=args.panoptic_segmentation,
        num_classes=num_classes,
    )
    learner.download()
    bounding_box_list = learner.infer(img)
    cv2.imshow('Detections', draw(img, bounding_box_list))
    cv2.waitKey(0)
