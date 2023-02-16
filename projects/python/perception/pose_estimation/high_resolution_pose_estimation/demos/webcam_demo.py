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
import time

from opendr.perception.pose_estimation import HighResolutionPoseEstimationLearner
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.pose_estimation import draw
import argparse
import numpy as np


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img_ = self.cap.read()
        if not was_read:
            raise StopIteration
        return img_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=True,
                        action="store_true")
    parser.add_argument("--height1", help="Base height of resizing in first inference, defaults to 420", default=420)
    parser.add_argument("--height2", help="Base height of resizing in second inference, defaults to 360", default=360)
    parser.add_argument("--input", help="use 'cam' for local webcam input or 'ip' for ip camera input, "
                                        "such as mobile phone with an appropriate application, defaults to 'cam'", 
                        default='ip')
    args = parser.parse_args()

    onnx, device, accelerate = args.onnx, args.device, args.accelerate
    base_height1, base_height2, input_ = args.height1, args.height2, args.input

    if accelerate:
        stride = True
        stages = 1
        half_precision = True
    else:
        stride = False
        stages = 2
        half_precision = False

    lw_pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                                   mobilenet_use_stride=stride, half_precision=half_precision)

    hr_pose_estimator = HighResolutionPoseEstimationLearner(device=device, num_refinement_stages=stages,
                                                            mobilenet_use_stride=stride, half_precision=half_precision,
                                                            first_pass_height=base_height1,
                                                            second_pass_height=base_height2,
                                                            percentage_around_crop=0.1)
    hr_pose_estimator.download(path=".", verbose=True)
    hr_pose_estimator.load("openpose_default")

    if onnx:
        hr_pose_estimator.optimize()

    lw_pose_estimator.download(path=".", verbose=True)
    lw_pose_estimator.load("openpose_default")

    if onnx:
        lw_pose_estimator.optimize()

    # Use the first camera available on the system
    if input_ == 'cam':
        image_provider = VideoReader(0)
    elif input_ == 'ip':
        image_provider = VideoReader('http://192.168.1.1:8080/video')  # use a local ip for an ip camera
        # e.g mobile application  "IP Camera Lite"

    hr_avg_fps = 0
    lw_avg_fps = 0

    image_provider = iter(image_provider)

    height = image_provider.cap.get(4)
    width = image_provider.cap.get(3)
    if width / height == 16 / 9:
        size = (1280, int(720 / 2))
    elif width / height == 4 / 3:
        size = (1024, int(768 / 2))
    else:
        size = (width, int(height / 2))

    while True:
        img = next(image_provider)

        total_time0 = time.time()
        img_copy = np.copy(img)

        # Perform inference
        start_time = time.perf_counter()
        hr_poses, heatmap = hr_pose_estimator.infer(img)
        hr_time = time.perf_counter() - start_time

        # Perform inference
        start_time = time.perf_counter()
        lw_poses = lw_pose_estimator.infer(img_copy)
        lw_time = time.perf_counter() - start_time

        total_time = time.time() - total_time0

        for hr_pose in hr_poses:
            draw(img, hr_pose)
        for lw_pose in lw_poses:
            draw(img_copy, lw_pose)

        lw_fps = 1 / (total_time - hr_time)
        hr_fps = 1 / (total_time - lw_time)
        # Calculate a running average on FPS
        hr_avg_fps = 0.95 * hr_avg_fps + 0.05 * hr_fps
        lw_avg_fps = 0.95 * lw_avg_fps + 0.05 * lw_fps

        cv2.putText(img=img, text="OpenDR High Resolution", org=(20, int(height / 10)),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=int(np.ceil(height / 600)), color=(200, 0, 0),
                    thickness=int(np.ceil(height / 600)))
        cv2.putText(img=img, text="Pose Estimation", org=(20, int(height / 10) + 50),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=int(np.ceil(height / 600)), color=(200, 0, 0),
                    thickness=int(np.ceil(height / 600)))

        cv2.putText(img=img_copy, text='Lightweight OpenPose ', org=(20, int(height / 10)),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=int(np.ceil(height / 600)), color=(200, 0, 0),
                    thickness=int(np.ceil(height / 600)))

        heatmap = heatmap * 5
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.resize(heatmap, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
        img[(img.shape[0] - heatmap.shape[0]):img.shape[0], 0:heatmap.shape[1]] = heatmap

        output_image = cv2.hconcat([img_copy, img])
        output_image = cv2.resize(output_image, size)
        cv2.imshow('Result', output_image)

        key = cv2.waitKey(1)
        if key == 27:
            exit(0)
