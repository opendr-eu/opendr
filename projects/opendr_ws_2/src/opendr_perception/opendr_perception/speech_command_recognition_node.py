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

import argparse
import torch
import numpy as np

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioData
from vision_msgs.msg import Classification2D

from opendr_bridge import ROS2Bridge
from opendr.engine.data import Timeseries
from opendr.perception.speech_recognition import MatchboxNetLearner, EdgeSpeechNetsLearner, QuadraticSelfOnnLearner


class SpeechRecognitionNode(Node):

    def __init__(self, input_audio_topic="/audio", output_speech_command_topic="/opendr/speech_recognition",
                 buffer_size=1.5, model="matchboxnet", model_path=None, device="cuda"):
        """
        Creates a ROS2 Node for speech command recognition
        :param input_audio_topic: Topic from which the audio data is received
        :type input_audio_topic: str
        :param output_speech_command_topic: Topic to which the predictions are published
        :type output_speech_command_topic: str
        :param buffer_size: Length of the audio buffer in seconds
        :type buffer_size: float
        :param model: base speech command recognition model: matchboxnet or quad_selfonn
        :type model: str
        :param device: device for inference ("cpu" or "cuda")
        :type device: str

        """
        super().__init__("opendr_speech_command_recognition_node")

        self.publisher = self.create_publisher(Classification2D, output_speech_command_topic, 1)

        self.create_subscription(AudioData, input_audio_topic, self.callback, 1)

        self.bridge = ROS2Bridge()

        # Initialize the internal audio buffer
        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((1, 1))

        # Initialize the recognition model
        if model == "matchboxnet":
            self.learner = MatchboxNetLearner(output_classes_n=20, device=device)
            load_path = "./MatchboxNet"
        elif model == "edgespeechnets":
            self.learner = EdgeSpeechNetsLearner(output_classes_n=20, device=device)
            assert model_path is not None, "No pretrained EdgeSpeechNets model available for download"
        elif model == "quad_selfonn":
            self.learner = QuadraticSelfOnnLearner(output_classes_n=20, device=device)
            load_path = "./QuadraticSelfOnn"

        # Download the recognition model
        if model_path is None:
            self.learner.download_pretrained(path=".")
            self.learner.load(load_path)
        else:
            self.learner.load(model_path)

        self.get_logger().info("Speech command recognition node started!")

    def callback(self, msg_data):
        """
        Callback that processes the input data and publishes predictions to the output topic
        :param msg_data: incoming message
        :type msg_data: audio_common_msgs.msg.AudioData
        """
        # Accumulate data until the buffer is full
        data = np.reshape(np.frombuffer(msg_data.data, dtype=np.int16)/32768.0, (1, -1))
        self.data_buffer = np.append(self.data_buffer, data, axis=1)

        if self.data_buffer.shape[1] > 16000*self.buffer_size:

            # Convert sample to OpenDR Timeseries and perform classification
            input_sample = Timeseries(self.data_buffer)
            class_pred = self.learner.infer(input_sample)

            # Publish output
            ros_class = self.bridge.from_category_to_rosclass(class_pred, self.get_clock().now().to_msg())
            self.publisher.publish(ros_class)

            # Reset the audio buffer
            self.data_buffer = np.zeros((1, 1))


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_audio_topic", type=str, default="/audio",
                        help="Listen to input data on this topic")
    parser.add_argument("-o", "--output_speech_command_topic", type=str, default="/opendr/speech_recognition",
                        help="Topic name for speech command output")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (cpu, cuda)")
    parser.add_argument("--buffer_size", type=float, default=1.5, help="Size of the audio buffer in seconds")
    parser.add_argument("--model", default="matchboxnet", choices=["matchboxnet", "edgespeechnets", "quad_selfonn"],
                        help="Model to be used for prediction: matchboxnet, edgespeechnets or quad_selfonn")
    parser.add_argument("--model_path", type=str,
                        help="Path to the model files, if not given, the pretrained model will be downloaded")
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

    speech_node = SpeechRecognitionNode(input_audio_topic=args.input_audio_topic,
                                        output_speech_command_topic=args.output_speech_command_topic,
                                        buffer_size=args.buffer_size, model=args.model, model_path=args.model_path,
                                        device=device)

    rclpy.spin(speech_node)

    speech_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
