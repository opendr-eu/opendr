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
import os
import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Classification2D

from opendr_bridge import ROS2Bridge
from opendr.engine.data import Image
from opendr.perception.multimodal_human_centric import RgbdHandGestureLearner


class RgbdHandGestureNode(Node):

    def __init__(self, input_rgb_image_topic="/kinect2/qhd/image_color_rect",
                 input_depth_image_topic="/kinect2/qhd/image_depth_rect",
                 output_gestures_topic="/opendr/gestures", device="cuda", delay=0.1):
        """
        Creates a ROS2 Node for gesture recognition from RGBD. Assuming that the following drivers have been installed:
        https://github.com/OpenKinect/libfreenect2 and https://github.com/code-iai/iai_kinect2.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param input_depth_image_topic: Topic from which we are reading the input depth image
        :type input_depth_image_topic: str
        :param output_gestures_topic: Topic to which we are publishing the predicted gesture class
        :type output_gestures_topic: str
        :param device: Device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param delay: Define the delay (in seconds) with which rgb message and depth message can be synchronized
        :type delay: float
        """
        super().__init__("opendr_rgbd_hand_gesture_recognition_node")

        self.gesture_publisher = self.create_publisher(Classification2D, output_gestures_topic, 1)

        image_sub = message_filters.Subscriber(self, ROS_Image, input_rgb_image_topic, qos_profile=1)
        depth_sub = message_filters.Subscriber(self, ROS_Image, input_depth_image_topic, qos_profile=1)
        # synchronize image and depth data topics
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=10, slop=delay)
        ts.registerCallback(self.callback)

        self.bridge = ROS2Bridge()

        # Initialize the gesture recognition
        self.gesture_learner = RgbdHandGestureLearner(n_class=16, architecture="mobilenet_v2", device=device)
        model_path = './mobilenet_v2'
        if not os.path.exists(model_path):
            self.gesture_learner.download(path=model_path)
        self.gesture_learner.load(path=model_path)

        # mean and std for preprocessing, based on HANDS dataset
        self.mean = np.asarray([0.485, 0.456, 0.406, 0.0303]).reshape(1, 1, 4)
        self.std = np.asarray([0.229, 0.224, 0.225, 0.0353]).reshape(1, 1, 4)

        self.get_logger().info("RGBD gesture recognition node started!")

    def callback(self, rgb_data, depth_data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param rgb_data: input image message
        :type rgb_data: sensor_msgs.msg.Image
        :param depth_data: input depth image message
        :type depth_data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image and preprocess
        rgb_image = self.bridge.from_ros_image(rgb_data, encoding='bgr8')
        depth_data.encoding = 'mono16'
        depth_image = self.bridge.from_ros_image_to_depth(depth_data, encoding='mono16')
        img = self.preprocess(rgb_image, depth_image)

        # Run gesture recognition
        gesture_class = self.gesture_learner.infer(img)

        #  Publish results
        ros_gesture = self.bridge.from_category_to_rosclass(gesture_class, self.get_clock().now().to_msg())
        self.gesture_publisher.publish(ros_gesture)

    def preprocess(self, rgb_image, depth_image):
        """
        Preprocess rgb_image, depth_image and concatenate them
        :param rgb_image: input RGB image
        :type rgb_image: engine.data.Image
        :param depth_image: input depth image
        :type depth_image: engine.data.Image
        """
        rgb_image = rgb_image.convert(format='channels_last') / (2**8 - 1)
        depth_image = depth_image.convert(format='channels_last') / (2**16 - 1)

        # resize the images to 224x224
        rgb_image = cv2.resize(rgb_image, (224, 224))
        depth_image = cv2.resize(depth_image, (224, 224))

        # concatenate and standardize
        img = np.concatenate([rgb_image, np.expand_dims(depth_image, axis=-1)], axis=-1)
        img = (img - self.mean) / self.std
        img = Image(img, dtype=np.float32)
        return img


def main(args=None):
    rclpy.init(args=args)

    # Default topics are according to kinectv2 drivers at https://github.com/OpenKinect/libfreenect2
    # and https://github.com/code-iai-iai_kinect2
    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/kinect2/qhd/image_color_rect")
    parser.add_argument("-id", "--input_depth_image_topic", help="Topic name for input depth image",
                        type=str, default="/kinect2/qhd/image_depth_rect")
    parser.add_argument("-o", "--output_gestures_topic", help="Topic name for predicted gesture class",
                        type=str, default="/opendr/gestures")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda",
                        choices=["cuda", "cpu"])
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

    gesture_node = RgbdHandGestureNode(input_rgb_image_topic=args.input_rgb_image_topic,
                                       input_depth_image_topic=args.input_depth_image_topic,
                                       output_gestures_topic=args.output_gestures_topic, device=device,
                                       delay=args.delay)

    rclpy.spin(gesture_node)

    gesture_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
