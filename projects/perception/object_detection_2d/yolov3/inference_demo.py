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

import cv2

from opendr.perception.object_detection_2d.yolov3.yolov3_learner import YOLOv3DetectorLearner
from opendr.perception.object_detection_2d.utils.vis_utils import draw_bounding_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    yolo = YOLOv3DetectorLearner(device=args.device)
    yolo.download(".", mode="pretrained")
    yolo.load("./yolo_default", verbose=True)

    yolo.download(".", mode="images", verbose=True)
    img = cv2.imread("./cat.jpg")

    boxes = yolo.infer(img)
    draw_bounding_boxes(img, boxes, class_names=yolo.classes, show=True)
