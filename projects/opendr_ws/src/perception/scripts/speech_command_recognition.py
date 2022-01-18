#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020-2022 OpenDR European Project
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

import rospy
import torch
import numpy as np
from opendr_bridge import ROSBridge
from audio_common_msgs.msg import AudioData
from vision_msgs.msg import Classification2D
import argparse

from opendr.engine.data import Timeseries
from opendr.perception.speech_recognition import MatchboxNetLearner, EdgeSpeechNetsLearner, QuadraticSelfOnnLearner


class SpeechRecognitionNode:

    def __init__(self, input_topic='/audio/audio', prediction_topic="/opendr/speech_recognition",
                 buffer_size=1.5, model='matchboxnet', model_path=None, device='cuda'):
        """
        Creates a ROS Node for speech command recognition
        :param input_topic: Topic from which the audio data is received
        :type input_topic: str
        :param prediction_topic: Topic to which the predictions are published
        :type prediction_topic: str
        :param buffer_size: Length of the audio buffer in seconds
        :type buffer_size: float
        :param model: base speech command recognition model: matchboxnet or quad_selfonn
        :type model: str
        :param device: device for inference ('cpu' or 'cuda')
        :type device: str

        """

        self.publisher = rospy.Publisher(prediction_topic, Classification2D, queue_size=10)

        rospy.Subscriber(input_topic, AudioData, self.callback)

        self.bridge = ROSBridge()

        # Initialize the internal audio buffer

        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((1, 1))

        # Initialize the recognition model
        if model == "matchboxnet":
            self.learner = MatchboxNetLearner(output_classes_n=20, device=device)
            load_path = './MatchboxNet'
        elif model == "edgespeechnets":
            self.learner = EdgeSpeechNetsLearner(output_classes_n=20, device=device)
            assert model_path is not None, "No pretrained EdgeSpeechNets model available for download"
        elif model == "quad_selfonn":
            self.learner = QuadraticSelfOnnLearner(output_classes_n=20, device=device)
            load_path = './QuadraticSelfOnn'

        # Download the recognition model
        if model_path is None:
            self.learner.download_pretrained(path='.')
            self.learner.load(load_path)
        else:
            self.learner.load(model_path)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_speech_command_recognition', anonymous=True)
        rospy.loginfo("Speech command recognition node started!")
        rospy.spin()

    def callback(self, msg_data):
        """
        Callback that processes the input data and publishes predictions to the output topic
        :param data: incoming message
        :type data: audio_common_msgs.msg.AudioData
        """
        # Accumulate data until the buffer is full
        data = np.reshape(np.frombuffer(msg_data.data, dtype=np.int16)/32768.0, (1, -1))
        self.data_buffer = np.append(self.data_buffer, data, axis=1)
        if self.data_buffer.shape[1] > 16000*self.buffer_size:

            # Convert sample to OpenDR Timeseries and perform classification
            input_sample = Timeseries(self.data_buffer)
            class_pred = self.learner.infer(input_sample)

            # Publish output
            ros_class = self.bridge.from_category_to_rosclass(class_pred)
            self.publisher.publish(ros_class)

            # Reset the audio buffer
            self.data_buffer = np.zeros((1, 1))


if __name__ == '__main__':
    # Select the device for running
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('input_topic', type=str, help='listen to input data on this topic')
    parser.add_argument('--buffer_size', type=float, default=1.5, help='size of the audio buffer in seconds')
    parser.add_argument('--model', choices=["matchboxnet", "edgespeechnets", "quad_selfonn"], default="matchboxnet",
                        help='model to be used for prediction: matchboxnet or quad_selfonn')
    parser.add_argument('--model_path', type=str,
                        help='path to the model files, if not given, the pretrained model will be downloaded')
    args = parser.parse_args()

    speech_node = SpeechRecognitionNode(input_topic=args.input_topic, buffer_size=args.buffer_size,
                                        model=args.model, model_path=args.model_path, device=device)
    speech_node.listen()
