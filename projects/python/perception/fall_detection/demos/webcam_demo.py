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
import time

import cv2

from opendr.perception.fall_detection import FallDetectorLearner
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.pose_estimation import draw, get_bbox


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
    parser.add_argument("--draw", help="Whether to draw additional pose lines", default=False, action="store_true")
    args = parser.parse_args()

    draw_poses = args.draw

    pose_estimator = LightweightOpenPoseLearner(device=args.device, num_refinement_stages=2,
                                                mobilenet_use_stride=False,
                                                half_precision=False)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    fall_detector = FallDetectorLearner(pose_estimator)

    image_provider = VideoReader(0)

    counter, fps = 0, 0
    try:
        for img in image_provider:

            start_time = time.perf_counter()
            detections = fall_detector.infer(img)
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)

            # Wait a few frames for FPS to stabilize
            if counter > 5:
                image = cv2.putText(img, "FPS: %.2f" % (fps,), (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (255, 255, 255), 1, cv2.LINE_AA)

            for detection in detections:
                fallen = detection[0].data
                pose = detection[1]

                if draw_poses:
                    draw(img, pose)

                if fallen == 1:
                    color = (0, 0, 255)
                    x, y, w, h = get_bbox(pose)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, "Detected fallen person", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1, cv2.LINE_AA)
                elif fallen == -1:
                    color = (0, 255, 0)
                    x, y, w, h = get_bbox(pose)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, "Detected standing person", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1, cv2.LINE_AA)
                elif fallen == 0:
                    color = (255, 255, 255)
                    x, y, w, h = get_bbox(pose)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, "Unable to detect if fallen", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1, cv2.LINE_AA)
            cv2.imshow('Result', img)
            cv2.waitKey(1)
            counter += 1
    except Exception as e:
        print(e)
        print("Inference fps: ", fps)
