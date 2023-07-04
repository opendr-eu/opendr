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

from typing import Optional
import wave
import argparse
from queue import Queue
from threading import Thread
from tempfile import NamedTemporaryFile

import numpy as np

import torch

import rospy
from audio_common_msgs.msg import AudioData
from opendr_bridge.msg import OpenDRTranscription

from opendr.perception.speech_transcription import (
    WhisperLearner,
    VoskLearner,
)
from opendr_bridge import ROSBridge
from opendr.engine.target import WhisperTranscription, VoskTranscription


class SpeechTranscriptionNode:
    def __init__(
        self,
        backbone: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        language: Optional[str] = None,
        download_dir: Optional[str] = None,
        input_audio_topic: str = "/audio/audio",
        output_transcription_topic: str = "/opendr/speech_transcription",
        node_topic: str = "opendr_transcription_node",
        device: str = "cuda",
        sample_width: int = 2,
        framerate: int = 16000,
    ):
        rospy.init_node(node_topic)

        self.data_queue = Queue()

        self.node_topic = node_topic
        self.backbone = backbone
        self.model_name = model_name
        self.model_path = model_path
        self.language = language
        self.download_dir = download_dir if download_dir is not None else "./"
        self.device = device
        self.sample_width = sample_width
        self.framerate = framerate

        # Initialize model
        if self.backbone == "whisper":
            name = self.model_path if self.model_path is not None else self.model_name
            assert name is not None, "Please specify a model name or path for Whisper."
            # Load Whisper model
            self.audio_model = WhisperLearner(language=self.language)
            self.audio_model.load(name=name, download_dir=self.download_dir)
        else:
            # Load Vosk model
            self.audio_model = VoskLearner()
            self.audio_model.load(
                name=self.model_name,
                model_path=self.model_path,
                language=self.language,
                download_dir=self.download_dir,
            )

        self.subscriber = rospy.Subscriber(input_audio_topic, AudioData, self.callback)
        self.publisher = rospy.Publisher(
            output_transcription_topic, OpenDRTranscription, queue_size=10
        )

        self.bridge = ROSBridge()

        self.temp_file = NamedTemporaryFile().name
        self.last_sample = b""
        self.cut_audio = False
        self.n_sample = None

        # Start processing thread
        self.processing_thread = Thread(target=self.process_audio)
        self.processing_thread.start()


    def callback(self, data):
        # Add data to queue
        self.data_queue.put(data.data)

    def _postprocess_whisper(self, audio_array: np.ndarray, transcription: WhisperTranscription) -> OpenDRTranscription:
        segments = transcription.segments

        if len(segments) > 1 and segments[-1]["text"] != "":
            # Detect ending a phrase.
            print("Detect ending a phrase.")
            last_segment = segments[-1]
            start_timestamp = last_segment["start"]
            self.n_sample = int(self.framerate * start_timestamp)
            self.cut_audio = True

            text = [
                segments[i]["text"].strip()
                for i in range(len(segments) - 1)
            ]
            text = " ".join(text)
            vosk_transcription = VoskTranscription(
                text=text, accept_waveform=True
            )
        elif self.cut_audio:
            # Long period of silence.
            print("Long period of silence.")
            self.n_sample = audio_array.shape[0]
            text = transcription.text.strip()
            vosk_transcription = VoskTranscription(
                text=text, accept_waveform=True
            )
        else:
            text = transcription.text.strip()
            vosk_transcription = VoskTranscription(
                text=text, accept_waveform=False
            )

        print(text)
        ros_transcription = self.bridge.to_ros_transcription(vosk_transcription)

        return ros_transcription

    def process_audio(self):
        while not rospy.is_shutdown():
            # Check if there is any data in the queue
            if not self.data_queue.empty():
                # Get the audio data from the queue
                # Concatenate our current audio data with the latest audio data.
                new_data = b""
                while not self.data_queue.empty():
                    new_data += self.data_queue.get()

                # print(self.cut_audio)
                if self.backbone == "vosk":
                    self.last_sample = new_data
                else:
                    self.last_sample += new_data

                # Write wav data to the temporary file.
                with wave.open(self.temp_file, "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(self.sample_width)
                    f.setframerate(self.framerate)
                    # Convert audio data to numpy array
                    numpy_data = np.frombuffer(self.last_sample, dtype=np.int16)
                    if self.backbone == "whisper" and self.cut_audio:
                        if self.n_sample is not None:
                            if self.n_sample < numpy_data.shape[0]:
                                if self.n_sample >= 3200:
                                    numpy_data = numpy_data[(self.n_sample - 3200) :]
                                else:
                                    numpy_data = numpy_data[(self.n_sample) :]
                                self.last_sample = self.last_sample[
                                    (self.n_sample * 2) :
                                ]
                                self.n_sample = None
                                print("--------------------------")

                        self.cut_audio = False

                    f.writeframes(numpy_data.tobytes())

                # Process audio
                if self.backbone == "vosk":
                    wf = wave.open(self.temp_file, "rb")
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break

                        transcription = self.audio_model.infer(data)

                        if transcription.accept_waveform:
                            print(f"Text: {transcription.text}")
                        else:
                            print(f"Partial: {transcription.text}")

                        ros_transcription = self.bridge.to_ros_transcription(
                            transcription
                        )
                        self.publisher.publish(ros_transcription)
                else:
                    audio_array = WhisperLearner.load_audio(self.temp_file)
                    phrase_timeout = 2  # Seconds
                    if audio_array.shape[0] > phrase_timeout * self.framerate:
                        t = self.audio_model.infer(
                            audio_array[-phrase_timeout * self.framerate :]
                        )
                        if t.text == "" or t.segments[-1]["no_speech_prob"] > 0.5:
                            print(t.segments[-1]["no_speech_prob"])
                            self.cut_audio = True
                    transcription_whisper = self.audio_model.infer(audio_array)

                    ros_transcription = self._postprocess_whisper(
                        audio_array, transcription_whisper
                    )
                    self.publisher.publish(ros_transcription)


    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--backbone",
        default="whisper",
        help="backbone to use for audio processing. Options: whisper, vosk",
        choices=["whisper", "vosk"],
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="model to use for audio processing. Options: tiny, small, medium, large",
    )
    parser.add_argument("--model-path", default=None, help="path to model")
    parser.add_argument(
        "--download-dir", default=None, help="directory to download models to"
    )
    parser.add_argument(
        "--language", default="en", help="language to use for audio processing"
    )
    parser.add_argument(
        "--input-audio-topic",
        default="/audio/audio",
        help="name of the topic to subscribe",
    )
    parser.add_argument(
        "--output-transcription-topic",
        default="/opendr/speech_transcription",
        help="name of the topic to publish",
    )
    parser.add_argument(
        "--node-topic",
        default="opendr_transcription_node",
        help="name of the transcription ros node",
    )
    parser.add_argument(
        "--sample-width", type=int, default=2, help="sample width for audio data"
    )
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\" for Whisper and \"cpu\" for Vosk",
        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--framerate", type=int, default=16000, help="framerate for audio data"
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

    if args.backbone == "whisper" and args.framerate != 16000:
        print("Whisper only supports 16000 Hz. Using 16000 Hz instead.")
        framerate = 16000
    else:
        framerate = args.framerate


    try:
        node = SpeechTranscriptionNode(backbone=args.backbone,
            model_name=args.model_name,
            model_path=args.model_path,
            download_dir=args.download_dir,
            language=None if args.language.lower() == "none" else args.language,
            input_audio_topic=args.input_audio_topic,
            output_transcription_topic=args.output_transcription_topic,
            node_topic=args.node_topic,
            device=args.device,
            sample_width=args.sample_width,
            framerate=framerate,
        )
        node.spin()
    except rospy.ROSInterruptException:
        pass
