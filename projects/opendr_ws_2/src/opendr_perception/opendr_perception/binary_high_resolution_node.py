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
from time import perf_counter

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.perception.binary_high_resolution import BinaryHighResolutionLearner


class BinaryHighResolutionNode(Node):

    def __init__(self, input_rgb_image_topic="image_raw", output_heatmap_topic="/opendr/binary_hr_heatmap",
                 output_rgb_image_topic="/opendr/binary_hr_heatmap_visualization", performance_topic=None,
                 model_path=None, architecture="VGG_720p", device="cuda"):
        """
        Create a ROS2 Node for binary high resolution classification with Binary High Resolution.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_heatmap_topic: Topic to which we are publishing the heatmap in the form of a ROS image
        :type output_heatmap_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the heatmap image blended with the
        input image for visualization purposes
        :type output_rgb_image_topic: str
        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic:  str
        :param model_path: The path to the directory of a trained model
        :type model_path: str
        :param architecture: Architecture used on trained model (`VGG_720p` or `VGG_1080p`)
        :type architecture: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        super().__init__('opendr_binary_high_resolution_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_heatmap_topic is not None:
            self.heatmap_publisher = self.create_publisher(ROS_Image, output_heatmap_topic, 1)
        else:
            self.heatmap_publisher = None

        if output_rgb_image_topic is not None:
            self.visualization_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.visualization_publisher = None

        if performance_topic is not None:
            self.performance_publisher = self.create_subscription(Float32, performance_topic, 1)
        else:
            self.performance_publisher = None

        self.bridge = ROS2Bridge()

        # Initialize the semantic segmentation model
        self.learner = BinaryHighResolutionLearner(device=device, architecture=architecture)
        try:
            self.learner.load(model_path)
        except FileNotFoundError:
            if model_path is None or model_path == "test_model":
                raise TypeError("A trained model path should be provided in the --model_path /path/to/model/dir "
                                "argument. Please refer to /projects/python/perception/binary_high_resolution/"
                                "train_eval_demo.py and train a model for your purposes first.")
            elif model_path != "test_model":
                # Handle case where relative path is provided with '/' in front
                try:
                    self.learner.load(model_path[1:])
                except FileNotFoundError:
                    raise FileNotFoundError("A trained model could not be found in the provided path. Try specifying the path "
                                            "with opendr_ws_2 as root or provide an absolute path.")

        self.get_logger().info(f"Model loaded from specified path: {model_path}")
        self.get_logger().info("Binary High Resolution node started.")

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        if self.performance_publisher:
            start_time = perf_counter()
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        image = image.convert("channels_last")
        # Run learner to retrieve the OpenDR heatmap
        heatmap = self.learner.infer(image)

        if self.performance_publisher:
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)  # NOQA
            fps_msg = Float32()
            fps_msg.data = fps
            self.performance_publisher.publish(fps_msg)

        # Publish heatmap in the form of an image
        if self.heatmap_publisher is not None:
            self.heatmap_publisher.publish(self.bridge.to_ros_image(heatmap))

        # Publish heatmap color visualization blended with the input image
        if self.visualization_publisher is not None:
            # Blend heatmap with image
            heatmap = cv2.normalize(heatmap.data, None, 0, 1, cv2.NORM_MINMAX)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
            self.visualization_publisher.publish(self.bridge.to_ros_image(Image(blended), encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/image_raw")
    parser.add_argument("-o", "--output_heatmap_topic", help="Topic to which we are publishing the heatmap in the form "
                                                             "of a ROS image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/binary_hr_heatmap")
    parser.add_argument("-ov", "--output_rgb_image_topic", help="Topic to which we are publishing the heatmap image "
                                                                "blended with the input image for visualization purposes",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/binary_hr_heatmap_visualization")
    parser.add_argument("--performance_topic", help="Topic name for performance messages, disabled (None) by default",
                        type=str, default=None)
    parser.add_argument("-m", "--model_path", help="Path to the directory of the trained model",
                        type=str, default="test_model")
    parser.add_argument("-a", "--architecture", help="Architecture used for the trained model, either \"VGG_720p\" or "
                                                     "\"VGG_1080p\", defaults to \"VGG_720p\"",
                        type=str, default="VGG_720p", choices=["VGG_720p", "VGG_1080p"])
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

    binary_hr_node = BinaryHighResolutionNode(device=device,
                                              input_rgb_image_topic=args.input_rgb_image_topic,
                                              output_heatmap_topic=args.output_heatmap_topic,
                                              output_rgb_image_topic=args.output_rgb_image_topic,
                                              performance_topic=args.performance_topic,
                                              model_path=args.model_path,
                                              architecture=args.architecture)

    rclpy.spin(binary_hr_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    binary_hr_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
