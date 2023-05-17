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

import os
import argparse

import librosa
import numpy as np
import torch

from opendr.engine.data import Timeseries
from opendr.perception.speech_transcription import (
    WhisperLearner,
    VoskLearner,
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'true'):
        return True
    elif v.lower() in ('False', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # Select the device to perform inference on
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except:
        device = "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Path to the input file")
    parser.add_argument(
        "--model",
        choices=["whisper", "vosk"],
        required=True,
        help="model to be used for transcription: whisper or vosk",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="path to the model files, if not given, the pretrained model will be downloaded",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Specific name for Whisper model",
        choices=f"Available models name: ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language for the model",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        help="Path to the directory where the model will be downloaded",
    )
    parser.add_argument(
        "--builtin-transcribe",
        type=str2bool,
        help="Use the built-in transcribe function of the Whisper model",
    )

    args = parser.parse_args()

    # Create a learner
    if args.model == "whisper":
        learner = WhisperLearner(model_name=args.model_name, fp16=True)
    elif args.model == "vosk":
        learner = VoskLearner(model_name=args.model_name, language=args.language)
    else:
        raise ValueError("Invalid model")

    # Load or download the model
    learner.load(model_path=args.model_path, download_dir=args.download_dir)

    # Load the audio file and run speech command recognition
    audio_input, _ = librosa.load(args.input, sr=learner.sample_rate)
    data = Timeseries(np.expand_dims(audio_input, axis=0))
    if args.model == "whisper":
        result = learner.infer(data, builtin_transcribe=args.builtin_transcribe)
    else:
        result = learner.infer(data)

    print(f"The word is: {result.data}")
