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

import argparse
from opendr.perception.multimodal_human_centric import AudiovisualEmotionLearner
import os

parser = argparse.ArgumentParser()

parser.add_argument('-input_video', type=str, help='path to video file', required=True)
parser.add_argument('-input_audio', type=str, help='path to audio file', required=True)

args = parser.parse_args()

assert os.path.exists(args.input_video), 'Provided input video file does not exist'
assert os.path.exists(args.input_audio), 'Provided input audio file does not exist'

avlearner = AudiovisualEmotionLearner(device='cuda', fusion='ia', mod_drop='zerodrop')

avlearner.download('model')
avlearner.load('model')

audio, video = avlearner.load_inference_data(args.input_audio, args.input_video)
prediction = avlearner.infer(audio, video)
print(avlearner.pred_to_label(prediction))
