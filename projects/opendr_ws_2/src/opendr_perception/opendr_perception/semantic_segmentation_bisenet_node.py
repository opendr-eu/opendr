#!/usr/bin/env python
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
import numpy as np
import torch
import cv2
import colorsys

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.engine.target import Heatmap
from opendr.perception.semantic_segmentation import BisenetLearner


class BisenetNode(Node):

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw", output_heatmap_topic="/opendr/heatmap",
                 output_rgb_image_topic="/opendr/heatmap_visualization", device="cuda"):
        """
        Creates a ROS2 Node for semantic segmentation with Bisenet.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_heatmap_topic: Topic to which we are publishing the heatmap in the form of a ROS image containing
        class ids
        :type output_heatmap_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the heatmap image blended with the
        input image and a class legend for visualization purposes
        :type output_rgb_image_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        super().__init__('opendr_semantic_segmentation_bisenet_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_heatmap_topic is not None:
            self.heatmap_publisher = self.create_publisher(ROS_Image, output_heatmap_topic, 1)
        else:
            self.heatmap_publisher = None

        if output_rgb_image_topic is not None:
            self.visualization_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.visualization_publisher = None

        self.bridge = ROS2Bridge()

        # Initialize the semantic segmentation model
        self.learner = BisenetLearner(device=device)
        self.learner.download(path="bisenet_camvid")
        self.learner.load("bisenet_camvid")

        self.class_names = ["Bicyclist", "Building", "Car", "Column Pole", "Fence", "Pedestrian", "Road", "Sidewalk",
                            "Sign Symbol", "Sky", "Tree", "Unknown"]
        self.colors = self.getDistinctColors(len(self.class_names))  # Generate n distinct colors

        self.get_logger().info("Semantic segmentation bisenet node initialized.")

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        try:
            # Run semantic segmentation to retrieve the OpenDR heatmap
            heatmap = self.learner.infer(image)

            # Publish heatmap in the form of an image containing class ids
            if self.heatmap_publisher is not None:
                heatmap = Heatmap(heatmap.data.astype(np.uint8))  # Convert to uint8
                self.heatmap_publisher.publish(self.bridge.to_ros_image(heatmap))

            # Publish heatmap color visualization blended with the input image and a class color legend
            if self.visualization_publisher is not None:
                heatmap_colors = Image(self.colors[heatmap.numpy()])
                image = Image(cv2.resize(image.convert("channels_last", "bgr"), (960, 720)))
                alpha = 0.4  # 1.0 means full input image, 0.0 means full heatmap
                beta = (1.0 - alpha)
                image_blended = cv2.addWeighted(image.opencv(), alpha, heatmap_colors.opencv(), beta, 0.0)
                # Add a legend
                image_blended = self.addLegend(image_blended, np.unique(heatmap.data))

                self.visualization_publisher.publish(self.bridge.to_ros_image(Image(image_blended),
                                                                              encoding='bgr8'))
        except Exception:
            self.get_logger().warn('Failed to generate prediction.')

    def addLegend(self, image, unique_class_ints):
        # Text setup
        origin_x, origin_y = 5, 5  # Text origin x, y
        color_rectangle_size = 25
        font_size = 1.0
        font_thickness = 2
        w_max = 0
        for i in range(len(unique_class_ints)):
            text = self.class_names[unique_class_ints[i]]  # Class name
            x, y = origin_x, origin_y + i * color_rectangle_size  # Text position
            # Determine class color and convert to regular integers
            color = (int(self.colors[unique_class_ints[i]][0]),
                     int(self.colors[unique_class_ints[i]][1]),
                     int(self.colors[unique_class_ints[i]][2]))
            # Get text width and height
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
            if w >= w_max:
                w_max = w
            # Draw partial background rectangle
            image = cv2.rectangle(image, (x - origin_x, y),
                                  (x + origin_x + color_rectangle_size + w_max,
                                   y + color_rectangle_size),
                                  (255, 255, 255, 0.5), -1)
            # Draw color rectangle
            image = cv2.rectangle(image, (x, y),
                                  (x + color_rectangle_size, y + color_rectangle_size), color, -1)
            # Draw class name text
            image = cv2.putText(image, text, (x + color_rectangle_size + 2, y + h),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness)
        return image

    @staticmethod
    def HSVToRGB(h, s, v):
        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        return np.array([int(255 * r), int(255 * g), int(255 * b)])

    def getDistinctColors(self, n):
        huePartition = 1.0 / (n + 1)
        return np.array([self.HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]).astype(np.uint8)


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="image_raw")
    parser.add_argument("-o", "--output_heatmap_topic", help="Topic to which we are publishing the heatmap in the form "
                                                             "of a ROS image containing class ids",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/heatmap")
    parser.add_argument("-ov", "--output_rgb_image_topic", help="Topic to which we are publishing the heatmap image "
                                                                "blended with the input image and a class legend for "
                                                                "visualization purposes",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/heatmap_visualization")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
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
                               output_heatmap_topic=args.output_heatmap_topic,
                               output_rgb_image_topic=args.output_rgb_image_topic)

    rclpy.spin(bisenet_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bisenet_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
