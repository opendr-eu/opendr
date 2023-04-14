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
import threading
import time
from typing import Callable
from logging import getLogger

import numpy as np
import sounddevice as sd

from opendr.perception.speech_recognition import WhisperLearner


logger = getLogger(__name__)


def record_audio(duration: int, sample_rate: int) -> np.ndarray:
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # Wait for the recording to finish
    audio_data = np.squeeze(audio_data, axis=-1)

    return audio_data


def transcribe_audio(audio_data: np.ndarray, transcribe_function: Callable, details: bool):
    print("Transcribing...")
    output = transcribe_function(audio_data)

    if not details:
        output = output["text"]

    print("Transcription: ", output)


def main(duration, sample_rate, interval, model_path, model_name, load_path, device, details):
    # Initialize the WhisperLearner class and load the model
    learner = WhisperLearner(model_name=model_name, device=device)
    learner.load(
        download_root=model_path,
        load_path=load_path,
    )

    while True:
        # Start a new thread for recording audio
        record_thread = threading.Thread(
            target=record_audio, args=(duration, sample_rate)
        )
        record_thread.start()

        # Start a new thread for transcribing the recorded audio
        transcribe_thread = threading.Thread(
            target=transcribe_audio,
            args=(record_audio(duration, sample_rate), learner.infer, details),
        )
        transcribe_thread.start()

        # Wait for both threads to finish
        record_thread.join()
        transcribe_thread.join()

        # Wait for `interval` seconds before starting the next recording
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record and transcribe audio every X seconds using the WhisperLearner model."
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5,
        help="Duration of the recording in seconds.",
    )
    parser.add_argument(
        "-s",
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate of the recording.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=10.0,
        help="Time interval between recordings in seconds.",
    )
    parser.add_argument(
        "-de",
        "--device",
        type=str,
        default="cpu",
        help="Device for running inference.",
    )
    parser.add_argument(
        "-l",
        "--load_path",
        type=str,
        required=False,
        help="Path to the pretrained Whisper model.",
    )
    parser.add_argument(
        "-p",
        "--download_path",
        type=str,
        required=False,
        help="Download path for the pretrained Whisper model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tiny.en",
        help="Name of the pretrained Whisper model.",
    )
    parser.add_argument(
        "--details",
        type=bool,
        required=False,
        default=False,
        help="Return the command with side information",
    )


    args = parser.parse_args()

    main(
        args.duration,
        args.sample_rate,
        args.interval,
        model_path=args.download_path,
        model_name=args.model_name,
        load_path=args.load_path,
        device=args.device,
        details=args.details,
    )
