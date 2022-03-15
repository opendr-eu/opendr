# Copyright 2020-2022 OpenDR European Project
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

from opendr.engine.data import Image
from opendr.perception.fall_detection import FallDetectorLearner
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.pose_estimation import draw


def fall_detection_on_img(img, draw_pose=False, draw_fall_detection_lines=False):
    detections = fall_detector.infer(img)
    if len(detections) == 0:
        img = img.opencv()
        img = cv2.putText(img, "No detections", (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                          0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Results', img)
        cv2.waitKey(0)
    for detection in detections:
        fallen = detection[0].data
        keypoints = detection[1]
        if fallen == 1:
            print("Detected fallen person.")
        elif fallen == -1:
            print("Didn't detect fallen person.")
        else:
            print("Can't detect fall.")

        img = img.opencv()
        if draw_pose:
            poses = pose_estimator.infer(img)
            for pose in poses:
                draw(img, pose)

        text = "CAN'T DETECT FALL"
        color = (255, 255, 255)
        if fallen == 1:
            text = "FALLEN"
            color = (0, 0, 255)
        elif fallen == -1:
            text = "STANDING"
            color = (0, 255, 0)

        img = cv2.putText(img, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                          0.75, color, 2, cv2.LINE_AA)

        if draw_fall_detection_lines:
            if keypoints[0].data[0] != -1:
                cv2.line(img, (int(keypoints[0].x), int(keypoints[0].y)),
                         (int(keypoints[1].x), int(keypoints[1].y)), color, 4)
            if keypoints[2].data[0] != -1:
                cv2.line(img, (int(keypoints[1].x), int(keypoints[1].y)),
                         (int(keypoints[2].x), int(keypoints[2].y)), color, 4)
        cv2.imshow('Results', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--draw", help="Whether to draw additional pose lines", default=False, action="store_true")
    args = parser.parse_args()

    onnx, device, draw_pose = args.onnx, args.device, args.draw
    stride = False
    stages = 2
    half_precision = False

    pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                                mobilenet_use_stride=stride, half_precision=half_precision)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")
    fall_detector = FallDetectorLearner(pose_estimator=pose_estimator)

    # TODO test images should be downloaded from FTP
    print("Running detector on image without humans")
    fall_detection_on_img(Image.open("fall_detection_images/no_humans.png"), draw_pose, draw_pose)
    print("Running detector on image fallen")
    fall_detection_on_img(Image.open("fall_detection_images/rgb_1240.png"), draw_pose, draw_pose)
    print("Running detector on image standing")
    fall_detection_on_img(Image.open("fall_detection_images/rgb_0088.png"), draw_pose, draw_pose)
