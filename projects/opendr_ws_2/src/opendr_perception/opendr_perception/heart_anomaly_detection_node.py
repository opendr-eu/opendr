#!/usr/bin/env python
# -*- coding: utf-8 -*-_
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

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Classification2D
from std_msgs.msg import Float32MultiArray

from opendr_bridge import ROS2Bridge
from opendr.perception.heart_anomaly_detection import GatedRecurrentUnitLearner, AttentionNeuralBagOfFeatureLearner


class HeartAnomalyNode(Node):

    def __init__(self, input_ecg_topic="/ecg/ecg", output_heart_anomaly_topic="/opendr/heart_anomaly",
                 device="cuda", model="anbof"):
        """
        Creates a ROS2 Node for heart anomaly (atrial fibrillation) detection from ecg data
        :param input_ecg_topic: Topic from which we are reading the input array data
        :type input_ecg_topic: str
        :param output_heart_anomaly_topic: Topic to which we are publishing the predicted class
        :type output_heart_anomaly_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model: model to use: anbof or gru
        :type model: str
        """
        super().__init__("opendr_heart_anomaly_detection_node")

        self.publisher = self.create_publisher(Classification2D, output_heart_anomaly_topic, 1)

        self.subscriber = self.create_subscription(Float32MultiArray, input_ecg_topic, self.callback, 1)

        self.bridge = ROS2Bridge()

        # AF dataset
        self.channels = 1
        self.series_length = 9000

        if model == 'gru':
            self.learner = GatedRecurrentUnitLearner(in_channels=self.channels, series_length=self.series_length,
                                                     n_class=4, device=device)
        elif model == 'anbof':
            self.learner = AttentionNeuralBagOfFeatureLearner(in_channels=self.channels, series_length=self.series_length,
                                                              n_class=4, device=device, attention_type='temporal')

        self.learner.download(path='.', fold_idx=0)
        self.learner.load(path='.')

        self.get_logger().info("Heart anomaly detection node initialized.")

    def callback(self, msg_data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param msg_data: input message
        :type msg_data: std_msgs.msg.Float32MultiArray
        """
        # Convert Float32MultiArray to OpenDR Timeseries
        data = self.bridge.from_rosarray_to_timeseries(msg_data, self.channels, self.series_length)

        # Run ecg classification
        class_pred = self.learner.infer(data)

        # Publish results
        ros_class = self.bridge.from_category_to_rosclass(class_pred, self.get_clock().now().to_msg())
        self.publisher.publish(ros_class)


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_ecg_topic", type=str, default="/ecg/ecg",
                        help="listen to input ECG data on this topic")
    parser.add_argument("-o", "--output_heart_anomaly_topic", type=str, default="/opendr/heart_anomaly",
                        help="Topic name for heart anomaly detection topic")
    parser.add_argument("--model", type=str, default="anbof", help="model to be used for prediction: anbof or gru",
                        choices=["anbof", "gru"])
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu, cuda)",
                        choices=["cuda", "cpu"])
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

    heart_anomaly_detection_node = HeartAnomalyNode(input_ecg_topic=args.input_ecg_topic,
                                                    output_heart_anomaly_topic=args.output_heart_anomaly_topic,
                                                    model=args.model, device=device)

    rclpy.spin(heart_anomaly_detection_node)

    heart_anomaly_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
