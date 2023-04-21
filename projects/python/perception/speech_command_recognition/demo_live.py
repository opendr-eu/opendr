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


from typing import Callable
import argparse
import time
from logging import getLogger

import numpy as np
import sounddevice as sd

from opendr.perception.speech_recognition import WhisperLearner


logger = getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'true'):
        return True
    elif v.lower() in ('False', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    output = output[0]

    if not details:
        output = output["text"]

    print("Transcription: ", output)

    return output


def wait_for_start_command(learner, sample_rate):
    print("Waiting for 'Hi Whisper' command...")
    print("Stop by saying 'Bye Whisper'.")
    while True:
        audio_data = record_audio(1, sample_rate)
        transcription = learner.infer(audio_data)[0]["text"].lower()
        print(f"User said: {transcription}")
        if "hi whisper" in transcription:
            print("Start command received. Starting the loop.")
            break
        time.sleep(1)


def main(duration, interval, model_path, model_name, load_path, device, details):
    # Initialize the WhisperLearner class and load the model
    learner = WhisperLearner(model_name=model_name, device=device)
    learner.load(
        download_root=model_path,
        load_path=load_path,
    )

    # Wait for the user to say "hi whisper" before starting the loop
    sample_rate = 16000
    wait_for_start_command(learner, sample_rate)

    while True:
        # Record the audio
        audio_data = record_audio(duration, sample_rate)

        # Transcribe the recorded audio and check for the "bye whisper" command
        transcription = transcribe_audio(audio_data, learner.infer, details)
        if not isinstance(transcription, str):
            transcription = transcription["text"]

        if "bye whisper" in transcription.lower():
            print("Stop command received. Exiting the program.")
            break

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
        "-i",
        "--interval",
        type=float,
        default=10.0,
        help="Time interval between recordings in seconds.",
    )
    parser.add_argument(
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
        default=".",
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
        type=str2bool,
        required=False,
        default=False,
        help="Return the command with side information",
    )


    args = parser.parse_args()

    main(
        duration=args.duration,
        interval=args.interval,
        model_path=args.download_path,
        model_name=args.model_name,
        load_path=args.load_path,
        device=args.device,
        details=args.details,
    )
