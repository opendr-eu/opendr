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
import time
# from opendr.perception.pose_estimation import LightweightOpenPoseLearner
# from opendr.perception.pose_estimation import HighResolutionPoseEstimationLearner
# from opendr.perception.pose_estimation import draw
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
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

    def getsize(self):
        cap = cv2.VideoCapture(self.file_name)
        w = cap.get(3)
        h = cap.get(4)
        return w, h


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=True,
                        action="store_true")
    parser.add_argument("--height1", help="Base height of resizing in first inference", default=420)
    parser.add_argument("--height2", help="Base height of resizing in second inference", default=360)
    parser.add_argument("--input", help="use cam for webcam input or ip for moblie phone input", default='cam')
    args = parser.parse_args()

    onnx, device = args.onnx, args.device
    accelerate = args.accelerate

    onnx, device, accelerate, base_height1, base_height2, input = args.onnx, args.device, args.accelerate, \
        args.height1, args.height2, args.input
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
    if input == 'cam':
        image_provider = VideoReader(0)
    elif input == 'ip':
        image_provider = VideoReader('http://155.207.108.144:8081/video')   # use a local ip for an ip camera
        # e.g mobile application  "IP Camera Lite"

    # un-comment and change path for saving the video
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('/home/thodoris/Desktop/odr_video_demo/clean/webcam_output' + str(base_height1) + '_' + str(
    #     base_height2) + '.avi', fourcc, 20.0, (1280, int(720 / 2)))
    hr_avg_fps = 0
    lw_avg_fps = 0

    image_provider = iter(image_provider)
    while True:
        img = next(image_provider)

        total_time0 = time.time()
        img_copy = np.copy(img)
        height, width, _ = img.shape

        # Perform inference
        start_time = time.perf_counter()
        hr_poses, xmin, ymin, xmax, ymax, heatmap = hr_pose_estimator.infer(img)
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

        cv2.putText(img=img, text='OpenDR High Resolution Pose Estimation ', org=(20, 50),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(200, 0, 0), thickness=1)

        cv2.putText(img=img_copy, text='Lightweight OpenPose ', org=(20, 50),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1,
                    color=(200, 0, 0), thickness=1)

        heatmap = heatmap * 5
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.resize(heatmap, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
        img[(img.shape[0] - heatmap.shape[0]):img.shape[0], 0:heatmap.shape[1]] = heatmap

        output_image = cv2.hconcat([img_copy, img])
        output_image = cv2.resize(output_image, (1280, int(720 / 2)))
        # uncomment next line for saving video
        # out.write(output_image)

        cv2.imshow('Result', output_image)

        key = cv2.waitKey(1)
        if key == 27:
            exit(0)
