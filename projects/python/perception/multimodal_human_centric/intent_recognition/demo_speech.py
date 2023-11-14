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
import os
import numpy as np
import sounddevice as sd

from opendr.perception.speech_transcription import (
    WhisperLearner,
    VoskLearner,
)
from opendr.perception.multimodal_human_centric import IntentRecognitionLearner

logger = getLogger(__name__)

LABELS = [
            'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
            'Agree', 'Taunt', 'Flaunt', 'Joke', 'Oppose', 'Comfort',
            'Care', 'Inform', 'Advise', 'Arrange', 'Introduce',
            'Leave', 'Prevent', 'Greet', 'Ask for help'
        ]


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


def transcribe_audio(audio_data: np.ndarray, transcribe_function: Callable):
    output = transcribe_function(audio_data)
    output = output.text

    print("Transcription: ", output)

    return output


def wait_for_start_command(learner, sample_rate):
    backbone = "Whisper" if isinstance(learner, WhisperLearner) else "Vosk"
    print(f"Waiting for 'Hi {backbone}' command...")
    print(f"Stop by saying 'Bye {backbone}'.")
    while True:
        audio_data = record_audio(1, sample_rate)
        transcription = learner.infer(audio_data).text.lower()
        print(f"User said: {transcription}")

        if isinstance(learner, WhisperLearner) and ("hi whisper" in transcription or "hi, whisper" in transcription):
            print("Start command received. Starting the loop.")
            break

        if isinstance(learner, VoskLearner) and ("hi vosk" in transcription or "hi, vosk" in transcription):
            print("Start command received. Starting the loop.")
            break
        time.sleep(1)


def get_intent_learner(text_backbone, device, cache_path, download_dir):
    if text_backbone == 'bert-small':
        text_backbone = 'prajjwal1/bert-small'
    elif text_backbone == 'bert-mini':
        text_backbone = 'prajjwal1/bert-mini'
    elif text_backbone == 'bert-tiny':
        text_backbone = 'prajjwal1/bert-tiny'

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    learner = IntentRecognitionLearner(text_backbone=text_backbone, mode='language',
                                       device=device, log_path='logs', results_path='results',
                                       output_path='outputs', cache_path=cache_path)
    if not os.path.exists('{}/{}.pth'.format(download_dir, args.text_backbone)):
        learner.download('{}/{}.pth'.format(download_dir, args.text_backbone))
    learner.load('{}/{}.pth'.format(download_dir, args.text_backbone))

    return learner


def main(backbone, text_backbone, duration, interval, model_path, model_name, language, download_dir, device, cache_path):

    if backbone == "whisper":
        if model_path is not None:
            name = model_path
        else:
            name = model_name
        learner = WhisperLearner(language=language, device=device)
        learner.load(name=name, download_dir=download_dir)
    elif args.backbone == "vosk":
        learner = VoskLearner()
        learner.load(
            name=model_name,
            model_path=model_path,
            language=language,
            download_dir=download_dir,
        )
    else:
        raise ValueError("invalid backbone")

    intent_learner = get_intent_learner(text_backbone, device, cache_path, download_dir)

    # Wait for the user to say "hi whisper" before starting the loop
    sample_rate = 16000
    wait_for_start_command(learner, sample_rate)

    while True:
        # Record the audio
        audio_data = record_audio(duration, sample_rate)

        # Transcribe the recorded audio and check for the "bye whisper" command
        transcription = transcribe_audio(audio_data, learner.infer).lower()
        prediction = intent_learner.infer({'text': transcription}, modality='language')
        print('Intent: ', [(LABELS[p.data], p.confidence) for p in prediction])
        if backbone == "whisper" and ("bye whisper" in transcription or "bye, whisper" in transcription):
            print("Stop command received. Exiting the program.")
            break

        if backbone == "vosk" and ("bye vosk" in transcription or "bye, whisper" in transcription):
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
        "--backbone",
        default="whisper",
        help="backbone to use for audio processing. Options: whisper, vosk",
        choices=["whisper", "vosk"],
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="path to the model files, if not given, the pretrained model will be downloaded",
        default=None,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Specific name for Whisper model, available models are: ['tiny.en', 'tiny', 'base.en', 'base', \
                 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']",
        choices=['tiny.en', 'tiny', 'base.en', 'base', 'small.en',
                 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large'],
        default="base.en",
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
        default="./pretrained_models"
    )
    parser.add_argument("--text_backbone", help="Text backbone: ['bert-base-uncased' | \
                        'albert-base-v2' | 'bert-small' | 'bert-mini' | 'bert-tiny']",
                        type=str, default="bert-base-uncased")
    parser.add_argument("--cache_path", help="cache path", type=str, default="cache")

    args = parser.parse_args()

    main(
        backbone=args.backbone,
        text_backbone=args.text_backbone,
        duration=args.duration,
        interval=args.interval,
        model_path=args.model_path,
        model_name=args.model_name,
        language=args.language,
        download_dir=args.download_dir,
        device=args.device,
        cache_path=args.cache_path
    )
