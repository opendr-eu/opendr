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

import time

import cv2

from opendr.engine.data import Image
from opendr.perception.fall_detection import FallDetectorLearner
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.pose_estimation import draw


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


def fall_detection_on_img(img):
    fallen, keypoints = fall_detector.infer(img)
    if fallen:
        print("Detected fallen person.")
    else:
        print("Didn't detect fallen person.")

    poses = pose_estimator.infer(img)
    img = img.opencv()
    for pose in poses:
        draw(img, pose)

    text = "CAN'T DETECT FALL"
    color = (255, 255, 255)
    if fallen.data == 1:
        text = "FALLEN"
        color = (0, 0, 255)
    elif fallen.data == 0:
        text = "STANDING"
        color = (0, 255, 0)

    img = cv2.putText(img, text, (0, 12), cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, color, 1, cv2.LINE_AA)

    if keypoints[0].data[0] != -1:
        cv2.line(img, (int(keypoints[0].data[0]), int(keypoints[0].data[1])),
                 (int(keypoints[1].data[0]), int(keypoints[1].data[1])), color, 4)
    if keypoints[2].data[0] != -1:
        cv2.line(img, (int(keypoints[1].data[0]), int(keypoints[1].data[1])),
                 (int(keypoints[2].data[0]), int(keypoints[2].data[1])), color, 4)
    cv2.imshow('Results', img)
    cv2.waitKey(0)


device = "cuda"
stride = False
stages = 2
half_precision = False

webcam = True
# TODO test images should be downloaded from FTP
image_path_fallen = "fall_detection_images/rgb_1240.png"
image_path_standing = "fall_detection_images/rgb_0088.png"

pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                            mobilenet_use_stride=stride, half_precision=half_precision)

pose_estimator.download(path="./", verbose=True)
pose_estimator.load("openpose_default")
fall_detector = FallDetectorLearner(pose_estimator=pose_estimator)

if not webcam:
    img_fallen = Image.open(image_path_fallen)
    img_standing = Image.open(image_path_standing)

    print("Running detector on image fallen")
    fall_detection_on_img(img_fallen)
    print("Running detector on image standing")
    fall_detection_on_img(img_standing)

else:
    image_provider = VideoReader(0)

    counter, avg_fps = 0, 0
    try:
        for img in image_provider:

            start_time = time.perf_counter()

            # Perform inference
            poses = pose_estimator.infer(img)
            fallen, keypoints = fall_detector.infer(img)

            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)

            for pose in poses:
                draw(img, pose)

            # Calculate a running average on FPS
            avg_fps = 0.8 * fps + 0.2 * fps

            # Wait a few frames for FPS to stabilize
            if counter > 5:
                image = cv2.putText(img, "FPS: %.2f" % (avg_fps,), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)

            text = "CAN'T DETECT FALL"
            color = (255, 255, 255)
            if fallen.data == 1:
                text = "FALLEN"
                color = (0, 0, 255)
            elif fallen.data == 0:
                text = "STANDING"
                color = (0, 255, 0)

            img = cv2.putText(img, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                              1, color, 2, cv2.LINE_AA)
            if keypoints[0].data[0] != -1:
                cv2.line(img, (int(keypoints[0].data[0]), int(keypoints[0].data[1])),
                         (int(keypoints[1].data[0]), int(keypoints[1].data[1])), color, 4)
            if keypoints[2].data[0] != -1:
                cv2.line(img, (int(keypoints[1].data[0]), int(keypoints[1].data[1])),
                         (int(keypoints[2].data[0]), int(keypoints[2].data[1])), color, 4)
            cv2.imshow('Result', img)
            cv2.waitKey(1)
            counter += 1
    except Exception as e:
        print(e)
        print("Average inference fps: ", avg_fps)
