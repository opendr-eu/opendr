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

import os
import argparse
import numpy as np
import torch
import librosa

import rospy
import message_filters
from sensor_msgs.msg import Image as ROS_Image
from audio_common_msgs.msg import AudioData
from vision_msgs.msg import Classification2D

from opendr_bridge import ROSBridge
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.avlearner import AudiovisualEmotionLearner
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm import spatial_transforms as transforms
from opendr.engine.data import Video, Timeseries


class AudiovisualEmotionNode:

    def __init__(self, input_video_topic="/usb_cam/image_raw", input_audio_topic="/audio/audio",
                 annotations_topic="/opendr/audiovisual_emotion", buffer_size=3.6, device="cuda"):
        """
        Creates a ROS Node for audiovisual emotion recognition
        :param input_video_topic: Topic from which we are reading the input video. Expects detected face of size 224x224
        :type input_video_topic: str
        :param input_audio_topic: Topic from which we are reading the input audio
        :type input_audio_topic: str
        :param annotations_topic: Topic to which we are publishing the predicted class
        :type annotations_topic: str
        :param buffer_size: length of audio and video in sec
        :type buffer_size: float
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """

        self.publisher = rospy.Publisher(annotations_topic, Classification2D, queue_size=10)

        video_sub = message_filters.Subscriber(input_video_topic, ROS_Image)
        audio_sub = message_filters.Subscriber(input_audio_topic, AudioData)
        # synchronize video and audio data topics
        ts = message_filters.ApproximateTimeSynchronizer([video_sub, audio_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        self.bridge = ROSBridge()

        # Initialize the gesture recognition
        self.avlearner = AudiovisualEmotionLearner(device=device, fusion='ia', mod_drop='zerodrop')
        if not os.path.exists('model'):
            self.avlearner.download('model')
        self.avlearner.load('model')

        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((1))
        self.video_buffer = np.zeros((1, 224, 224, 3))

        self.video_transform = transforms.Compose([
                              transforms.ToTensor(255)])

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_audiovisualemotion_recognition', anonymous=True)
        rospy.loginfo("Audiovisual emotion recognition node started!")
        rospy.spin()

    def callback(self, image_data, audio_data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param image_data: input image message, face image of size 224x224
        :type image_data: sensor_msgs.msg.Image
        :param audio_data: input audio message, speech
        :type audio_data: audio_common_msgs.msg.AudioData
        """
        audio_data = np.reshape(np.frombuffer(audio_data.data, dtype=np.int16)/32768.0, (1, -1))
        self.data_buffer = np.append(self.data_buffer, audio_data)
        image_data = self.bridge.from_ros_image(image_data, encoding='bgr8').convert(format='channels_last')
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
            ros_class = self.bridge.from_category_to_rosclass(class_pred)
            self.publisher.publish(ros_class)

            self.data_buffer = np.zeros((1))
            self.video_buffer = np.zeros((1, 224, 224, 3))


def select_distributed(m, n): return [i*n//m + n//(2*m) for i in range(m)]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_topic', type=str, help='listen to video input data on this topic')
    parser.add_argument('--audio_topic', type=str, help='listen to audio input data on this topic')
    parser.add_argument('--buffer_size', type=float, default=3.6, help='size of the audio buffer in seconds')
    args = parser.parse_args()

    avnode = AudiovisualEmotionNode(input_video_topic=args.video_topic, input_audio_topic=args.audio_topic,
                                    annotations_topic="/opendr/audiovisual_emotion",
                                    buffer_size=args.buffer_size, device=device)
    avnode.listen()
