#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import torch
import numpy as np
import os
import argparse
from opendr.perception.multimodal_human_centric import RgbdHandGestureLearner
from opendr.engine.data import Image
import cv2
import imageio

MEAN = np.asarray([0.485, 0.456, 0.406, 0.0303]).reshape(1, 1, 4)
STD = np.asarray([0.229, 0.224, 0.225, 0.0353]).reshape(1, 1, 4)
CLASS2CAT = {0: 'COLLAB',
             1: 'Eight',
             2: 'Five',
             3: 'Four',
             4: 'Horiz HBL, HFR',
             5: 'Horiz HFL, HBR',
             6: 'Nine',
             7: 'One',
             8: 'Punch',
             9: 'Seven',
             10: 'Six',
             11: 'Span',
             12: 'Three',
             13: 'TimeOut',
             14: 'Two',
             15: 'XSign'}

if __name__ == '__main__':
    # Select the device for running
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('-input_rgb', type=str, help='path to RGB image', required=True)
    parser.add_argument('-input_depth', type=str, help='path to Depth image', required=True)

    args = parser.parse_args()

    # create learner and load checkpoint
    gesture_learner = RgbdHandGestureLearner(n_class=16, architecture='mobilenet_v2', device=device)
    model_path = './mobilenet_v2'
    if not os.path.exists(model_path):
        gesture_learner.download(path=model_path)
    gesture_learner.load(path=model_path)

    rgb_image = imageio.imread(args.input_rgb)
    depth_image = imageio.imread(args.input_depth)

    # image preprocessing
    rgb_image = np.asarray(rgb_image) / (2**8 - 1)
    depth_image = np.asarray(depth_image) / (2**16 - 1)
    rgb_image = cv2.resize(rgb_image, (224, 224))
    depth_image = cv2.resize(depth_image, (224, 224))
    img = np.concatenate([rgb_image, np.expand_dims(depth_image, axis=-1)], axis=-1)
    img = (img - MEAN)/STD
    img = Image(img, dtype=np.float32)
    prediction = gesture_learner.infer(img)

    # convert numeric label to readable class label and print
    print(CLASS2CAT[prediction.data] + ' with confidence ' + str(prediction.confidence))
