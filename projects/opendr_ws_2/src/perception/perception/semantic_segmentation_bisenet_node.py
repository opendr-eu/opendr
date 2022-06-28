#!/usr/bin/env python
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

import rclpy
from rclpy.node import Node

import torch
import cv2
import numpy as np

from sensor_msgs.msg import Image as ROS_Image
from ros2_bridge.bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.perception.semantic_segmentation import BisenetLearner


class BisenetNode(Node):

    def __init__(self, input_image_topic="image_raw", output_heatmap_topic="opendr/heatmap", device="cuda"):
        super().__init__('semantic_segmentation_bisenet_node')

        self.input_image_topic = input_image_topic
        self.output_heatmap_topic = output_heatmap_topic

        if self.output_heatmap_topic is not None:
            self._heatmap_publisher = self.create_publisher(ROS_Image, output_heatmap_topic, 10)
        else:
            self._heatmap_publisher = None

        self.image_subscriber = self.create_subscription(ROS_Image, input_image_topic, self.callback, 10)

        self.bridge = ROS2Bridge()

        # Initialize the semantic segmentation model
        self._learner = BisenetLearner(device=device)
        self._learner.download(path="bisenet_camvid")
        self._learner.load("bisenet_camvid")
        
        self._colors = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    def callback(self, data):
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        # image = self.bridge.from_ros_image(data)
        cv2.imshow("image", image.opencv())
        cv2.waitKey(5)

        try:
            # Retrieve the OpenDR heatmap
            prediction = self._learner.infer(image)

            if self._heatmap_publisher is not None:
                heatmap_np = prediction.numpy()
                heatmap_o = self._colors[heatmap_np]
                heatmap_o = cv2.resize(np.uint8(heatmap_o), (960, 720))
                self._heatmap_publisher.publish(self.bridge.to_ros_image(Image(heatmap_o), encoding='bgr8'))
        except Exception:
            self.get_logger().warn('Failed to generate prediction.')


def main(args=None):
    rclpy.init(args=args)
    try:
        if torch.cuda.is_available():
            print("GPU found.")
            device = 'cuda'
        else:
            print("GPU not found. Using CPU instead.")
            device = 'cpu'
    except:
        device = 'cpu'

    bisenet_node = BisenetNode(device=device)

    rclpy.spin(bisenet_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bisenet_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
