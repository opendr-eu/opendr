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
from opendr.perception.pose_estimation import HighResolutionPoseEstimationLearner
from opendr.perception.pose_estimation import draw
from opendr.engine.data import Image
import argparse
from os.path import join


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    parser.add_argument("--height1", help="Base height of resizing in first inference", default=360)
    parser.add_argument("--height2", help="Base height of resizing in second inference", default=540)

    args = parser.parse_args()

    device, accelerate, base_height1, base_height2 = args.device, args.accelerate,\
        args.height1, args.height2

    if accelerate:
        stride = True
        stages = 0
        half_precision = True
    else:
        stride = False
        stages = 2
        half_precision = False

    pose_estimator = HighResolutionPoseEstimationLearner(device=device, num_refinement_stages=stages,
                                                         mobilenet_use_stride=stride, half_precision=half_precision,
                                                         first_pass_height=int(base_height1),
                                                         second_pass_height=int(base_height2))
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    # Download one sample image
    pose_estimator.download(path=".", mode="test_data")

    image_path = join("temp", "dataset", "image", "000000000785_1080.jpg")

    img = Image.open(image_path)

    poses = pose_estimator.infer(img)

    img_cv = img.opencv()
    for pose in poses:
        draw(img_cv, pose)
    cv2.imshow('Results', img_cv)
    cv2.waitKey(0)
