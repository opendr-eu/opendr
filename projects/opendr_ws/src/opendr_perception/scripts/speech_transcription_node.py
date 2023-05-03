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
import argparse

import torch

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from opendr.engine.data import Timeseries
from opendr.perception.speech_recognition import WhisperLearner
from opendr_bridge import ROSBridge
from opendr_bridge.msg import OpenDRTranscription


class SpeechTranscriptionNode:
    def __init__(
        self,
        input_audio_topic="/audio/audio",
        output_speech_command_topic="/opendr/speech_transcription",
        buffer_size=1.5,
        model="tiny.en",
        model_path=None,
        device="cuda",
    ):
        """
        Creates a ROS Node for speech transcription. 
        :param input_audio_topic: Topic from which the audio data is received
        :type input_audio_topic: str
        :param output_speech_command_topic: Topic to which the predictions are published
        :type output_speech_command_topic: str
        :param buffer_size: Length of the audio buffer in seconds
        :type buffer_size: float
        :param model: str
        :type model: str
        :param device: device for inference ("cpu" or "cuda")
        :type device: str

        """

        self.publisher = rospy.Publisher(
            output_speech_command_topic, OpenDRTranscription, queue_size=10
        )

        rospy.Subscriber(input_audio_topic, numpy_msg(Floats), self.callback)

        self.bridge = ROSBridge()

        # Initialize the internal audio buffer

        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((1, 1))

        # Initialize the transcription model
        self.learner = WhisperLearner(model_name=model, device=device)

        # Download the transcription model
        self.learner.load(load_path=model_path)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node("opendr_speech_transciption_node", anonymous=True)
        rospy.loginfo("Speech transcription node started.")
        rospy.spin()

    def callback(self, msg_data):
        """
        Callback that processes the input data and publishes transcription to the output topic
        :param msg_data: incoming message
        :type msg_data: rospy.numpy_msg.numpy_msg
        """
        # Accumulate data until the buffer is full
        self.data_buffer = np.append(self.data_buffer, msg_data.data)
        if self.data_buffer.shape[0] > 16000 * self.buffer_size:
            self.data_buffer = self.data_buffer.squeeze().astype(np.float32)
            x = self.data_buffer
            result = self.learner.infer(x)
            result = result[0] 

            # Publish output
            ros_transcription = self.bridge.to_ros_transcription(result)
            self.publisher.publish(ros_transcription)

            # Reset the audio buffer
            self.data_buffer = np.zeros((1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_audio_topic",
        type=str,
        default="audio/audio",
        help="Listen to input data on this topic",
    )
    parser.add_argument(
        "-o",
        "--output_speech_command_topic",
        type=str,
        default="/opendr/speech_transcription",
        help="Topic name for speech transcription output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cpu, cuda)",
    )
    parser.add_argument(
        "--buffer_size",
        type=float,
        default=1.5,
        help="Size of the audio buffer in seconds",
    )
    parser.add_argument(
        "--model",
        default="tiny.en",
        choices="Available models name: ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']",
        help="Model to be used for prediction: matchboxnet, edgespeechnets or quad_selfonn",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model files, if not given, the pretrained model will be downloaded",
    )
    args = parser.parse_args()

    try:
        if args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif args.device == "cuda":
            print("GPU not found. Using CPU instead.")
            device = "cpu"
        else:
            print("Using CPU")
            device = "cpu"
    except:
        print("Using CPU")
        device = "cpu"

    speech_node = SpeechTranscriptionNode(
        input_audio_topic=args.input_audio_topic,
        output_speech_command_topic=args.output_speech_command_topic,
        buffer_size=args.buffer_size,
        model=args.model,
        model_path=args.model_path,
        device=device,
    )
    speech_node.listen()
