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

from opendr.engine.data import Image
from opendr.perception.fall_detection import FallDetectorLearner
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.pose_estimation import draw, get_bbox


def fall_detection_on_img(img, draw_pose=False):
    detections = fall_detector.infer(img)
    img = img.opencv()

    if len(detections) == 0:
        print("- No detections.")
        cv2.putText(img, "No detections", (5, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Results', img)
        cv2.waitKey(0)
    else:
        for detection in detections:
            fallen = detection[0].data
            pose = detection[1]
            print("- Detected person.")
            if fallen == 1:
                print("  Detected fallen person.")
            elif fallen == -1:
                print("  Didn't detect fallen person.")
            else:
                print("  Can't detect fall.")

            if draw_pose:
                draw(img, pose)

            if fallen == 1:
                color = (0, 0, 255)
                x, y, w, h = get_bbox(pose)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, "Detected fallen person", (5, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)
            elif fallen == -1:
                color = (0, 255, 0)
                x, y, w, h = get_bbox(pose)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, "Detected standing person", (5, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)

        cv2.imshow('Results', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    args = parser.parse_args()

    pose_estimator = LightweightOpenPoseLearner(device=args.device, num_refinement_stages=2,
                                                mobilenet_use_stride=False,
                                                half_precision=False)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    fall_detector = FallDetectorLearner(pose_estimator)

    # Download a sample dataset
    fall_detector.download(".", verbose=True)

    print("Running detector on image without humans...")
    fall_detection_on_img(Image.open("test_images/no_person.png"))
    print("Running detector on image with a fallen person...")
    fall_detection_on_img(Image.open("test_images/fallen.png"))
    print("Running detector on image with a standing person...")
    fall_detection_on_img(Image.open("test_images/standing.png"))
