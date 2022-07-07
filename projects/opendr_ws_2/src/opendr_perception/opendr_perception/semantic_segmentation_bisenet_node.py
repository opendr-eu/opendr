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

import argparse
import torch
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROS_Image
from opendr_ros2_bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.perception.semantic_segmentation import BisenetLearner


class BisenetNode(Node):

    def __init__(self, input_rgb_image_topic="image_raw", output_rgb_image_topic="/opendr/heatmap", device="cuda"):
        """
        Initialize the Bisenet ROS node and create an instance of the respective learner class.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: ROS topic for the predicted heatmap
        :type output_rgb_image_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        super().__init__('semantic_segmentation_bisenet_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        self.heatmap_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)

        self.bridge = ROS2Bridge()

        # Initialize the semantic segmentation model
        self.learner = BisenetLearner(device=device)
        self.learner.download(path="bisenet_camvid")
        self.learner.load("bisenet_camvid")

        self.colors = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

        self.get_logger().info("Semantic segmentation bisenet node initialized.")

    def callback(self, data):
        """
        Predict the heatmap from the input image and publish the results.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        try:
            # Run semantic segmentation to retrieve the OpenDR heatmap
            heatmap = self.learner.infer(image)

            if self.heatmap_publisher is not None:
                heatmap_np = heatmap.numpy()
                heatmap_o = self.colors[heatmap_np]
                heatmap_o = cv2.resize(np.uint8(heatmap_o), (960, 720))
                self.heatmap_publisher.publish(self.bridge.to_ros_image(Image(heatmap_o), encoding='bgr8'))
        except Exception:
            self.get_logger().warn('Failed to generate prediction.')


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=str, default="/opendr/heatmap")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    try:
        if args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif args.device == "cuda":
            print("GPU not found. Using CPU instead.")
            device = "cpu"
        else:
            print("Using CPU.")
            device = "cpu"
    except:
        print("Using CPU.")
        device = "cpu"

    bisenet_node = BisenetNode(device=device,
                               input_rgb_image_topic=args.input_rgb_image_topic,
                               output_rgb_image_topic=args.output_rgb_image_topic)

    rclpy.spin(bisenet_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bisenet_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
