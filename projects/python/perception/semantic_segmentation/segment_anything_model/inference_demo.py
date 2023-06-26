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

import cv2
import argparse
import time
import matplotlib.pyplot as plt
from opendr.engine.data import Image
from opendr.perception.semantic_segmentation import SamLearner
from opendr.perception.object_detection_2d.yolov5.yolov5_learner import YOLOv5DetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Image path", type=str)
    parser.add_argument("-dc", "--detect_class", help="Class to segment out of yolov5 classes", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--show", help="Whether to open result image", default=False, action="store_true")
    args = parser.parse_args()

    yolo_detector = YOLOv5DetectorLearner(model_name="yolov5l6.pt", device="cpu")

    sam = SamLearner(device=args.device)

    img = cv2.imread(args.image)  # "../office_chair.jpg")

    odr_img = Image(img)

    start_time_full = time.perf_counter()

    detections = yolo_detector.infer(odr_img)
    draw_bounding_boxes(img, detections, yolo_detector.classes, line_thickness=3)

    blended_img = img
    for d in detections:
        print("found bounding box class:", yolo_detector.classes[int(d.name)])
        if yolo_detector.classes[int(d.name)] == args.detect_class:
            print("running segmentation on class:", args.detect_class)
            start_time = time.perf_counter()
            # Perform segmentation
            mask, scores, logits, bbox_prompt = sam.infer(img, d)

            blended_img = sam.draw(blended_img, bbox_prompt, mask)

            end_time = time.perf_counter()
            fps = round(1.0 / (end_time - start_time), 2)
            print(f"{round(end_time - start_time, 2)} seconds per sam inference, fps:{fps}")

    end_time_full = time.perf_counter()
    fps = round(1.0 / (end_time_full - start_time_full), 2)
    print(f"{round(end_time_full - start_time_full, 2)} seconds for detection+sam, fps:{fps}")

    if args.show:
        try:
            cv2.imshow('Result', blended_img)
        except:
            cv2.imshow('Result', img)
        cv2.waitKey(0)
