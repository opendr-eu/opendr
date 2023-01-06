#!/usr/bin/env python
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

from opendr.perception.object_detection_2d import GemLearner
from opendr.perception.object_detection_2d.gem.algorithm.util.draw import draw
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    # First we initialize the learner
    learner = GemLearner(num_classes=7, device=args.device)
    # Next, we download a pretrained model
    learner.download(mode='pretrained_gem')
    # And some sample images
    learner.download(mode='test_data_sample_images')
    # We now read the sample images
    m1_img = cv2.imread('temp/sample_images/rgb/2021_04_22_21_35_47_852516.jpg')
    m2_img = cv2.imread('temp/sample_images/aligned_infra/2021_04_22_21_35_47_852516.jpg')
    # Perform inference
    bounding_box_list, w_sensor1, _ = learner.infer(m1_img, m2_img)
    # Visualize the detections
    # The blue/green bar shows the contributions of the two modalities
    # Fully blue means relying purely on the first modality
    # Fully green means relying purely on the second modality
    cv2.imshow('Detections', draw(m1_img, bounding_box_list, w_sensor1))
    cv2.waitKey(0)
