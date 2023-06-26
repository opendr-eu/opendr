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


yolo_detector = YOLOv5DetectorLearner(model_name="yolov5l6.pt", device="cpu")

sam = SamLearner(device="cuda")

img = cv2.imread("../example.jpg")

start_time = time.perf_counter()
odr_img = Image(img)
detections = yolo_detector.infer(odr_img)
draw_bounding_boxes(img, detections, yolo_detector.classes, line_thickness=3)

blended_img = None
for d in detections:
    if yolo_detector.classes[int(d.name)] == "person":

        # Perform segmentation
        mask, scores, logits, bbox_prompt = sam.infer(img, d)

        blended_img = sam.draw(img, bbox_prompt, mask)

end_time = time.perf_counter()
fps = 1.0 / (end_time - start_time)

cv2.imshow('Result', blended_img)
cv2.waitKey(0)
print(fps)