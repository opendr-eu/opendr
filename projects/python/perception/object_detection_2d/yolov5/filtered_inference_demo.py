# Copyright 2020-2024 OpenDR European Project
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
import torch
import numpy as np

from opendr.engine.data import Image
from opendr.perception.object_detection_2d import YOLOv5DetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes

from opendr.perception.object_detection_2d.utils.class_filter_wrapper import FilteredLearnerWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--classes", help="Classes of interest to detect", type=str, nargs="+", default=['person'])
    args = parser.parse_args()

    yolo = YOLOv5DetectorLearner(model_name='yolov5s', device=args.device)

    for f in 'zidane.jpg', 'bus.jpg':
        torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
    im1 = Image.open('zidane.jpg')  # OpenDR image
    im2 = cv2.imread('bus.jpg')  # OpenCV image (BGR to RGB)

    # By default, only objects of the 'person' class are of interest
    filtered_yolo = FilteredLearnerWrapper(yolo, allowed_classes=args.classes)

    # detection before filtering
    print("No filtering...")
    results = yolo.infer(im1)
    draw_bounding_boxes(im1.opencv(), results, yolo.classes, show=True, line_thickness=3)

    # detection after filtering
    print("With filtering...")
    filtered_results = filtered_yolo.infer(im1)
    draw_bounding_boxes(im1.opencv(), filtered_results, filtered_yolo.classes, show=True, line_thickness=3)

    # detection before filtering
    print("No filtering...")
    results = yolo.infer(im2)
    draw_bounding_boxes(np.copy(im2), results, yolo.classes, show=True, line_thickness=3)

    # detection after filtering
    print("With filtering...")
    filtered_results = filtered_yolo.infer(im2)
    draw_bounding_boxes(np.copy(im2), filtered_results, filtered_yolo.classes, show=True, line_thickness=3)
