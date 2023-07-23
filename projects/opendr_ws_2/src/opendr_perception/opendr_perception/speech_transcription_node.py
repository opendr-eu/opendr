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


from typing import Tuple, Union, Optional
import wave
import argparse
import warnings
from time import perf_counter
from queue import Queue
from threading import Thread
from tempfile import NamedTemporaryFile

import numpy as np

import torch

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioData
from std_msgs.msg import Float32

from opendr_interface.msg import OpenDRTranscription
from opendr.perception.speech_transcription import (
    WhisperLearner,
    VoskLearner,
)

from opendr_bridge import ROS2Bridge
from opendr.engine.target import WhisperTranscription, VoskTranscription


def str2bool(string):
    str2val = {"True": True, "true": True, "False": False, "false": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_float(string):
    return None if string == "None" else float(string)


class SpeechTranscriptionNode(Node):
    def __init__(
        self,
        backbone: str,
        model_name: Optional[str]=None,
        model_path: Optional[str]=None,
        language: Optional[str]=None,
        download_dir: Optional[str]=None,
        temperature: Union[float, Tuple[float, ...]]=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        logprob_threshold: Optional[float]=-0.8,
        no_speech_threshold: float=0.6,
        phrase_timeout: float=2,
        input_audio_topic: str="/audio/audio",
        output_transcription_topic: str="/opendr/speech_transcription",
        performance_topic: Optional[str]=None,
        verbose: bool=False,
        device: str="cuda",
        sample_width: int=2,
        sample_rate: int=16000,
    ):
        """
        Creates a ROS2 Node for speech transcription using Whisper or Vosk.

        :param backbone: Backbone to use for audio processing. Options: whisper, vosk.
        :type backbone: str

        :param model_name: Model to use for audio processing. Options: tiny, tiny.en, base, base.en for Whisper,
        vosk-model-small-en-us-0.15 for Vosk.
        :type model_name: str

        :param model_path: Path to model files.
        :type model_path: str

        :param language: Language to use for audio processing. Example: 'en' for Whisper, 'en-us' for Vosk."
        :type language: str

        :param download_dir: Directory to download models to.
        :type download_dir: str

        :param temperature: Temperature to use for whisper decoding.
        :type temperature: Union[float, Tuple[float, ...]]

        :param logprob_threshold: Threshold for certainty in produced transcription.
        :type logprob_threshold: Optional[float]

        :param no_speech_threshold: Threshold for detecting long silence in Whisper.
        :type no_speech_threshold: float.

        :param phrase_timeout: The most recent seconds used for detecting long silence in Whisper.
        :type phrase_timeout: float.

        :param input_audio_topic: Name of the topic to subscribe.
        :type input_audio_topic: str.

        :param output_transcription_topic: Name of the topic to publish.
        :type output_transcription_topic: str.

        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic: str

        :param verbose: Display transcription.
        :type verbose: bool.

        :param device: Device to use, either 'cpu; or 'cuda' for Whisper and 'cpu' for Vosk.
        :type device: str.

        :param sample_width: Sample width to write audio data to wav file. Check your audio source for correct value.
        :type sample_width: int.

        :param sample_rate: Sampling rate for audio data. Check your audio source for correct value.
        :type sample_rate: int.
        """
        super().__init__("opendr_transcription_node")

        self.data_queue = Queue()

        self.backbone = backbone
        self.model_name = model_name
        self.model_path = model_path
        self.language = language
        self.download_dir = download_dir
        self.verbose = verbose
        self.device = device
        self.sample_width = sample_width
        self.sample_rate = sample_rate

        # Whisper parameters
        self.temperature = temperature
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.phrase_timeout = phrase_timeout

        # Initialize model
        if self.backbone == "whisper":
            # Load Whisper model
            self.audio_model = WhisperLearner(
                temperature=self.temperature,
                logprob_threshold=self.logprob_threshold,
                no_speech_threshold=self.no_speech_threshold,
                language=self.language,
                device=device,
            )
            self.audio_model.load(
                name=self.model_name,
                model_path=self.model_path,
                download_dir=self.download_dir,
            )
        else:
            # Load Vosk model
            self.audio_model = VoskLearner()
            self.audio_model.load(
                name=self.model_name,
                model_path=self.model_path,
                language=self.language,
                download_dir=self.download_dir,
            )

        self.create_subscription(AudioData, input_audio_topic, self.callback, 1)
        self.publisher = self.create_publisher(
             OpenDRTranscription, output_transcription_topic, 1
        )
        if performance_topic is not None:
            self.performance_publisher = self.create_publisher(Float32, performance_topic, 1)
        else:
            self.performance_publisher = None

        self.bridge = ROS2Bridge()

        self.temp_file = NamedTemporaryFile().name
        self.last_sample = b""
        self.cut_audio = False
        self.n_sample = None
        self.vad = None

        # Start processing thread
        self.processing_thread = Thread(target=self.process_audio)
        self.processing_thread.start()

    def callback(self, data):
        """
        Put data to queue.
        """
        self.data_queue.put(data.data)

    def _write_to_wav(self, numpy_data: np.ndarray):
        """
        Write numpy array to wav file.
        """
        # Write wav data to the temporary file.
        with wave.open(self.temp_file, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(self.sample_width)
            f.setframerate(self.sample_rate)
            # Convert audio data to numpy array

            f.writeframes(numpy_data.tobytes())

    def _vosk_preprocess_audio(self):
        """
        Get audio data from the queue, convert it to numpy array and write it to a wav file.
        """
        new_data = b""
        while not self.data_queue.empty():
            new_data += self.data_queue.get()

        self.last_sample = new_data  # Vosk operates on short utterances.

        # Convert audio data to numpy array
        numpy_data = np.frombuffer(self.last_sample, dtype=np.int16)

        self._write_to_wav(numpy_data)

    def _display_transcription(self, transcription: VoskTranscription):
        """
        Display the transcription

        :param transcription: Transcription from Vosk model.
        :type transcription: VoskTranscription.
        """
        if transcription.accept_waveform:
            print(f"Text: {transcription.text}")
        else:
            print(f"Partial: {transcription.text}")

    def _vosk_process_and_publish(self):
        """
        Process the audio data by Vosk and publish the transcription.
        """
        wf = wave.open(self.temp_file, "rb")
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            transcription = self.audio_model.infer(data)
            ros_transcription = self.bridge.to_ros_transcription(transcription)
            self.publisher.publish(ros_transcription)

            if self.verbose:
                self._display_transcription(transcription)

    def _whisper_preprocess_audio(self):
        """
        Store the audio from the queue and concatenate it with the previous audio if necessary.
        """
        while not self.data_queue.empty():
            self.last_sample += (
                self.data_queue.get()
            )  # Whisper operates on long sequence of text.

        if self.n_sample is not None:
            # The timestamp is not appropriate, longer than the audio.
            if self.n_sample * 2 > len(self.last_sample):
                pass
        if len(self.last_sample) < 3200:
            # Audio too short.
            pass
        else:
            if self.cut_audio:
                self.last_sample = self.last_sample[((self.n_sample - 1600) * 2):]
                self.cut_audio = False
                self.n_sample = None
            elif self.vad:
                self.last_sample = self.last_sample[((self.n_sample - 1600) * 2):]
                self.vad = False
                self.n_sample = None

        numpy_data = np.frombuffer(self.last_sample, dtype=np.int16)
        self._write_to_wav(numpy_data)

    def _whisper_vad(self, audio_array: np.ndarray) -> bool:
        """
        Voice activity detection. Detect long silence in the latest part of the
        audio. If silence is detected, reset the audio data.

        :param audio_array: Audio data for some recent seconds.
        :type audio_array: np.ndarray.
        """
        if audio_array.shape[0] > self.phrase_timeout * self.sample_rate:
            t = self.audio_model.infer(
                audio_array[-int(self.phrase_timeout * self.sample_rate):]
            )
            if t.text == "" or t.segments[-1]["no_speech_prob"] > self.no_speech_threshold:
                return True

        return False

    def _postprocess_whisper(
        self, audio_array: np.ndarray, transcription: WhisperTranscription
    ) -> VoskTranscription:
        """
        Detecting if a phrase in ended. Whisper does natively support this functionality.

        :param audio_array: Audio data.
        :type audio_array: np.ndarray.

        :param transcription: Transcription for Whisper learner inference.
        :type transcription: VoskTranscription
        """
        segments = transcription.segments

        accept_waveform = True
        if len(segments) > 1 and segments[-1]["text"] != "":
            if self.verbose:
                print("End of phrase detected.")
            last_segment = segments[-1]
            start_timestamp = last_segment["start"]
            self.n_sample = int(self.sample_rate * start_timestamp)
            self.cut_audio = True

            text = " ".join(
                [segments[i]["text"].strip() for i in range(len(segments) - 1)]
            )
        elif self.vad:
            if self.verbose:
                print("Long period of silence.")
            self.n_sample = audio_array.shape[0]
            text = transcription.text.strip()
        else:
            text = transcription.text.strip()
            accept_waveform = False

        vosk_transcription = VoskTranscription(
            text=text, accept_waveform=accept_waveform
        )

        return vosk_transcription

    def _whisper_process_and_publish(self):
        """
        Process the audio data by Whisper and publish the transcription.
        """
        audio_array = WhisperLearner.load_audio(self.temp_file)
        self.vad = self._whisper_vad(audio_array)
        transcription_whisper = self.audio_model.infer(audio_array)

        vosk_transcription = self._postprocess_whisper(
            audio_array, transcription_whisper
        )
        ros_transcription = self.bridge.to_ros_transcription(vosk_transcription)
        self.publisher.publish(ros_transcription)

        if self.verbose:
            self._display_transcription(vosk_transcription)

    def process_audio(self):
        """
        Process the audio data from the queue and publish the transcription.
        """
        while rclpy.ok():
            # Check if there is any data in the queue
            if not self.data_queue.empty():
                if self.performance_publisher:
                    start_time = perf_counter()

                # Process audio
                if self.backbone == "vosk":
                    self._vosk_preprocess_audio()
                    self._vosk_process_and_publish()
                else:
                    self._whisper_preprocess_audio()
                    self._whisper_process_and_publish()

                if self.performance_publisher:
                    end_time = perf_counter()
                    fps = 1.0 / (end_time - start_time)
                    fps_msg = Float32()
                    fps_msg.data = fps
                    self.performance_publisher.publish(fps_msg)


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        default="vosk",
        help="Backbone model for speech transcription. Options: vosk, whisper",
        choices=["vosk", "whisper"],
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Specific model name for each backbone. Example: tiny, tiny.en, base, base.en for Whisper,"
        "vosk-model-small-en-us-0.15 for Vosk.",
    )
    parser.add_argument("--model_path", default=None, help="Path to downloaded model files")
    parser.add_argument(
        "--download_dir", default=None, help="Directory to download models to"
    )
    parser.add_argument(
        "--language",
        default="en-us",
        help=(
            "Whisper uses the language parameter to avoid language detection, "
            "Vosk uses the language parameter to select a specific model. "
            "Example: 'en' for Whisper, 'en-us' for Vosk."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Temperature to use for whisper decoding."
    )
    parser.add_argument(
        "--temperature_increment_on_fallback",
        type=optional_float,
        default=0.2,
        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
    )
    parser.add_argument(
        "--logprob_threshold",
        default=-0.8,
        type=float,
        help="Threshold for certainty in produced transcription.",
    )
    parser.add_argument(
        "--phrase_timeout",
        default=2.0,
        type=float,
        help="The most recent seconds used for detecting long silence in Whisper.",
    )
    parser.add_argument(
        "--no_speech_threshold",
        default=0.6,
        type=float,
        help="Threshold for detecting long silence in Whisper.",
    )
    parser.add_argument(
        "-i", "--input_audio_topic",
        default="/audio/audio",
        help="Name of the topic to subscribe.",
    )
    parser.add_argument(
        "-o", "--output_transcription_topic",
        default="/opendr/speech_transcription",
        help="Name of the topic to publish.",
    )
    parser.add_argument(
        "--performance_topic",
        type=str,
        default=None,
        help="Topic name for performance messages, disabled (None) by default",
    )
    parser.add_argument(
        "--device",
        help="Device to use, either 'cpu; or 'cuda' for Whisper and 'cpu' for Vosk",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--verbose", default=False, type=str2bool, help="Display transcription."
    )
    parser.add_argument(
        "--sample_width", type=int, default=2, help="Sample width to write audio data to wav file."
        "Check your audio source for correct value."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Sampling rate for audio data."
        "Check your audio source for correct value."
    )
    args = parser.parse_args()

    try:
        if args.device == "cuda" and torch.cuda.is_available():
            if args.backbone == "whisper":
                device = "cuda"
            else:
                print("Vosk only supports CPU. Using CPU instead.")
                device = "cpu"
        elif args.device == "cuda":
            print("GPU not found. Using CPU instead.")
            device = "cpu"
        else:
            print("Using CPU.")
            device = "cpu"
    except:
        print("Using CPU.")
        device = "cpu"

    sample_rate = args.sample_rate
    if args.backbone == "whisper":
        if args.sample_rate != 16000:
            print("Whisper only supports 16000 Hz. Using 16000 Hz instead.")
            sample_rate = 16000

        if args.model_name.endswith(".en") and args.language not in {"en", "English"}:
            if args.language is not None:
                warnings.warn(
                    f"{args.model_name} is an English-only model but receipted language is '{args.language}';"
                    "using English instead."
                )
            args.language = "en"

    temperature = args.temperature
    increment = args.temperature_increment_on_fallback
    if increment is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    transcription_node = SpeechTranscriptionNode(
        backbone=args.backbone,
        model_name=args.model_name,
        model_path=args.model_path,
        download_dir=args.download_dir,
        language=None if args.language.lower() == "none" else args.language,
        phrase_timeout=args.phrase_timeout,
        temperature=temperature,
        logprob_threshold=args.logprob_threshold,
        no_speech_threshold=args.no_speech_threshold,
        input_audio_topic=args.input_audio_topic,
        output_transcription_topic=args.output_transcription_topic,
        performance_topic=args.performance_topic,
        verbose=args.verbose,
        device=device,
        sample_width=args.sample_width,
        sample_rate=sample_rate,
    )

    rclpy.spin(transcription_node)

    transcription_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
