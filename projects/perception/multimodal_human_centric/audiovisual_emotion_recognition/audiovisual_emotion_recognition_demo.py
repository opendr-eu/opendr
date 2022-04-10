#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import argparse
import librosa
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner./
    algorithm.data import get_audiovisual_emotion_dataset
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.avlearner import AudiovisualEmotionLearner

NUM_2_CLASS = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}
parser = argparse.ArgumentParser()

parser.add_argument('-input_video', type=str, help='path to video file', required=True)
parser.add_argument('-input_audio', type=str, help='path to audio file', required=True)

args = parser.parse_args()

avlearner = AudiovisualEmotionLearner(device='cuda', fusion='ia', mod_drop='zerodrop')

avlearner.download('model')
avlearner.load('model')

audio, video = avlearner.load_inference_data(args.input_audio, args.input_video)
prediction = avlearner.infer(audio, video)
print(NUM_2_CLASS[prediction.data])
