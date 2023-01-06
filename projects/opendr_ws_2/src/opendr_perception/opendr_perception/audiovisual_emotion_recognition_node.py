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
import numpy as np
import torch
import librosa
import cv2

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image as ROS_Image
from audio_common_msgs.msg import AudioData
from vision_msgs.msg import Classification2D

from opendr_bridge import ROS2Bridge
from opendr.perception.multimodal_human_centric import AudiovisualEmotionLearner
from opendr.perception.multimodal_human_centric import spatial_transforms as transforms
from opendr.engine.data import Video, Timeseries


class AudiovisualEmotionNode(Node):

    def __init__(self, input_video_topic="/image_raw", input_audio_topic="/audio",
                 output_emotions_topic="/opendr/audiovisual_emotion", buffer_size=3.6, device="cuda",
                 delay=0.1):
        """
        Creates a ROS2 Node for audiovisual emotion recognition
        :param input_video_topic: Topic from which we are reading the input video. Expects detected face of size 224x224
        :type input_video_topic: str
        :param input_audio_topic: Topic from which we are reading the input audio
        :type input_audio_topic: str
        :param output_emotions_topic: Topic to which we are publishing the predicted class
        :type output_emotions_topic: str
        :param buffer_size: length of audio and video in sec
        :type buffer_size: float
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param delay: Define the delay (in seconds) with which rgb message and depth message can be synchronized
        :type delay: float
        """
        super().__init__("opendr_audiovisual_emotion_recognition_node")

        self.publisher = self.create_publisher(Classification2D, output_emotions_topic, 1)

        video_sub = message_filters.Subscriber(self, ROS_Image, input_video_topic, qos_profile=1)
        audio_sub = message_filters.Subscriber(self, AudioData, input_audio_topic, qos_profile=1)
        # synchronize video and audio data topics
        ts = message_filters.ApproximateTimeSynchronizer([video_sub, audio_sub], queue_size=10, slop=delay,
                                                         allow_headerless=True)
        ts.registerCallback(self.callback)

        self.bridge = ROS2Bridge()

        self.avlearner = AudiovisualEmotionLearner(device=device, fusion='ia', mod_drop='zerodrop')
        if not os.path.exists('model'):
            self.avlearner.download('model')
        self.avlearner.load('model')

        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((1))
        self.video_buffer = np.zeros((1, 224, 224, 3))

        self.video_transform = transforms.Compose([
                              transforms.ToTensor(255)])

        self.get_logger().info("Audiovisual emotion recognition node started!")

    def callback(self, image_data, audio_data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param image_data: input image message, face image
        :type image_data: sensor_msgs.msg.Image
        :param audio_data: input audio message, speech
        :type audio_data: audio_common_msgs.msg.AudioData
        """
        audio_data = np.reshape(np.frombuffer(audio_data.data, dtype=np.int16)/32768.0, (1, -1))
        self.data_buffer = np.append(self.data_buffer, audio_data)

        image_data = self.bridge.from_ros_image(image_data, encoding='bgr8').convert(format='channels_last')
        image_data = cv2.resize(image_data, (224, 224))

        self.video_buffer = np.append(self.video_buffer, np.expand_dims(image_data.data, 0), axis=0)

        if self.data_buffer.shape[0] > 16000*self.buffer_size:
            audio = librosa.feature.mfcc(self.data_buffer[1:], sr=16000, n_mfcc=10)
            audio = Timeseries(audio)

            to_select = select_distributed(15, len(self.video_buffer)-1)
            video = self.video_buffer[1:][to_select]

            video = [self.video_transform(img) for img in video]
            video = Video(torch.stack(video, 0).permute(1, 0, 2, 3))

            class_pred = self.avlearner.infer(audio, video)

            # Publish output
            ros_class = self.bridge.from_category_to_rosclass(class_pred, self.get_clock().now().to_msg())
            self.publisher.publish(ros_class)

            self.data_buffer = np.zeros((1))
            self.video_buffer = np.zeros((1, 224, 224, 3))


def select_distributed(m, n): return [i*n//m + n//(2*m) for i in range(m)]


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-iv", "--input_video_topic", type=str, default="/image_raw",
                        help="Listen to video input data on this topic")
    parser.add_argument("-ia", "--input_audio_topic", type=str, default="/audio",
                        help="Listen to audio input data on this topic")
    parser.add_argument("-o", "--output_emotions_topic", type=str, default="/opendr/audiovisual_emotion",
                        help="Topic name for output emotions recognition")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cpu, cuda)", choices=["cuda", "cpu"])
    parser.add_argument("--buffer_size", type=float, default=3.6,
                        help="Size of the audio buffer in seconds")
    parser.add_argument("--delay", help="The delay (in seconds) with which RGB message and"
                        "depth message can be synchronized", type=float, default=0.1)

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

    emotion_node = AudiovisualEmotionNode(input_video_topic=args.input_video_topic,
                                          input_audio_topic=args.input_audio_topic,
                                          output_emotions_topic=args.output_emotions_topic,
                                          buffer_size=args.buffer_size, device=device, delay=args.delay)

    rclpy.spin(emotion_node)

    emotion_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
