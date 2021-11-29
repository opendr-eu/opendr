#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020-2021 OpenDR European Project
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
from opendr.engine.data import Timeseries
from vision_msgs.msg import Classification2D, ObjectHypothesis
import argparse
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from opendr_bridge import ROSBridge
import os
import message_filters
import cv2
from opendr.perception.heart_anomaly_detection.gated_recurrent_unit.gated_recurrent_unit_learner import \
    GatedRecurrentUnitLearner
from opendr.perception.heart_anomaly_detection.attention_neural_bag_of_feature\
    .attention_neural_bag_of_feature_learner import AttentionNeuralBagOfFeatureLearner


class HeartAnomalyNode:

    def __init__(self, input_topic="/ecg/ecg", prediction_topic="/opendr/heartanomaly", device="cuda", model='anbof'):
        """
        Creates a ROS Node for heart anomaly (atrial fibrillation) detection from ecg data
        :param input_topic: Topic from which we are reading the input array data
        :type input_topic: str
        :param prediction_topic: Topic to which we are publishing the predicted class
        :type prediction_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model: model to use: anbof or gru
        :type model: str
        """

        self.publisher = rospy.Publisher(prediction_topic, Classification2D, queue_size=10)

        rospy.Subscriber(input_topic, Float32MultiArray, self.callback)

        self.bridge = ROSBridge()

        # AF dataset
        self.channels = 1
        self.series_length = 9000

        # Initialize the gesture recognition
        if model == 'gru':
            self.learner = GatedRecurrentUnitLearner(in_channels=self.channels, series_length=self.series_length,
                    n_class=4, device=device, attention_type='temporal')
        elif model == 'anbof':
            self.learner = AttentionNeuralBagOfFeatureLearner(in_channels=self.channels, series_length=self.series_length, 
                    n_class=4, device=device, attention_type='temporal')

        self.learner.download(path='.', fold_idx=0)
        self.learner.load(path='.')

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_heart_anomaly_detection', anonymous=True)
        rospy.loginfo("Heart anomaly detection node started!")
        rospy.spin()

    def callback(self, msg_data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: std_msgs.msg.Float32MultiArray
        """
        # Convert Float32MultiArray to OpenDR Timeseries
        data = np.reshape(msg_data.data, (self.channels, self.series_length))
        data = Timeseries(data)

        # Run ecg classification
        class_pred = self.learner.infer(data)

        # Publish results
        ros_class = self.bridge.from_category_to_rosclass(class_pred)
        self.publisher.publish(ros_class)

if __name__ == '__main__':
    # Select the device for running
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('input_topic', type=str, help='listen to input data on this topic')
    parser.add_argument('model', type=str, help='model to be used for prediction: anbof or gru')
    args = parser.parse_args()

    gesture_node = HeartAnomalyNode(input_topic=args.input_topic, model=args.model, device=device)
    gesture_node.listen()
