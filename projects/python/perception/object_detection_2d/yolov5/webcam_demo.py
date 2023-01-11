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

import cv2
import time

from opendr.engine.data import Image
from opendr.perception.object_detection_2d import YOLOv5DetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--model", help="Model to use", type=str, default="yolov5s",
                        choices=['yolov5s', 'yolov5n', 'yolov5m', 'yolov5l', 'yolov5x',
                                 'yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'custom'])
    args = parser.parse_args()

    yolo_detector = YOLOv5DetectorLearner(model_name=args.model, device=args.device)

    # Use the first camera available on the system
    image_provider = VideoReader(0)
    fps = -1.0
    try:
        for img in image_provider:

            img = Image(img)
            start_time = time.perf_counter()
            detections = yolo_detector.infer(img)
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            img = img.opencv()
            if detections:
                draw_bounding_boxes(img, detections, yolo_detector.classes, line_thickness=3)

            img = cv2.putText(img, "FPS: %.2f" % (fps,), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Result', img)
            cv2.waitKey(1)
    except:
        print("Inference fps: ", round(fps))
