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

import numpy as np
import torch
import argparse
import librosa
from opendr.engine.data import Timeseries
from opendr.perception.speech_recognition import MatchboxNetLearner, EdgeSpeechNetsLearner, QuadraticSelfOnnLearner


if __name__ == '__main__':
    # Select the device to perform inference on
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help="Path to the input file")
    parser.add_argument('--model', choices=["matchboxnet", "edgespeechnets", "quad_selfonn"], required=True,
                        help='model to be used for prediction: matchboxnet or quad_selfonn')
    parser.add_argument('--model_path', type=str,
                        help='path to the model files, if not given, the pretrained model will be downloaded')
    parser.add_argument('--n_class', type=int, help='Number of classes', default=20)

    args = parser.parse_args()

    # Create a learner
    if args.model == "matchboxnet":
        learner = MatchboxNetLearner(output_classes_n=args.n_class, device=device)
        load_path = './MatchboxNet'
    elif args.model == "edgespeechnets":
        learner = EdgeSpeechNetsLearner(output_classes_n=args.n_class, device=device)
        assert args.model_path is not None, "No pretrained EdgeSpeechNets model available for download"
    elif args.model == "quad_selfonn":
        learner = QuadraticSelfOnnLearner(output_classes_n=args.n_class, device=device)
        load_path = './QuadraticSelfOnn'

    # Load or download the model
    if args.model_path is None:
        learner.download_pretrained(path='.')
        learner.load(load_path)
    else:
        learner.load(args.model_path)

    # Load the audio file and run speech command recognition
    audio_input, _ = librosa.load(args.input, sr=learner.sample_rate)
    data = Timeseries(np.expand_dims(audio_input, axis=0))
    result = learner.infer(data)
    print(result)
