#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020-2024 OpenDR European Project
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
import numpy as np

import torch

from opendr.engine.data import Timeseries
from opendr.perception.speech_transcription import (
    VoskLearner,
    WhisperLearner,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("True", "true"):
        return True
    elif v.lower() in ("False", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    # Select the device to perform inference on
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Path to the input file")
    parser.add_argument(
        "--backbone",
        default="whisper",
        help="backbone to use for audio processing. Options: whisper, vosk",
        choices=["whisper", "vosk"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the model files, if not given, the pretrained model will be downloaded",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Specific name for Whisper model",
        choices="Available models name: ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium',"
        "'large-v1', 'large-v2', 'large']",
        default=None,
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language for the model",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        help="Path to the directory where the model will be downloaded",
    )
    parser.add_argument(
        "--builtin_transcribe",
        type=str2bool,
        help="Use the built-in transcribe function of the Whisper model",
    )

    args = parser.parse_args()

    # Create a learner
    if args.backbone == "whisper":
        learner = WhisperLearner(language=args.language)
        learner.load(name=args.model_name, model_path=args.model_path, download_dir=args.download_dir)
    elif args.backbone == "vosk":
        learner = VoskLearner()
        learner.load(
            name=args.model_name,
            model_path=args.model_path,
            language=args.language,
            download_dir=args.download_dir,
        )
    else:
        raise ValueError("invalid backbone")

    # Load the audio file and run speech command recognition
    audio_input, _ = librosa.load(args.input, sr=learner.sample_rate)
    data = Timeseries(np.expand_dims(audio_input, axis=0))
    result = learner.infer(data)

    print(f"The transcription is: {result.text}")
