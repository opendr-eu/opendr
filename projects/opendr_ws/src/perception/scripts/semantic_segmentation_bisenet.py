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
import numpy as np
import torch
import cv2

import rospy
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge

from opendr.engine.data import Image
from opendr.perception.semantic_segmentation import BisenetLearner


class BisenetNode:

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw", output_rgb_image_topic="/opendr/heatmap",
                 device="cuda"):
        """
        Creates a ROS Node for semantic segmentation with Bisenet.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the heatmap image
        :type output_rgb_image_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        self.input_rgb_image_topic = input_rgb_image_topic

        if output_rgb_image_topic is not None:
            self.heatmap_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=1)
        else:
            self.heatmap_publisher = None

        self.bridge = ROSBridge()

        # Initialize the semantic segmentation model
        self.learner = BisenetLearner(device=device)
        self.learner.download(path="bisenet_camvid")
        self.learner.load("bisenet_camvid")

        self.colors = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('semantic_segmentation_bisenet_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Semantic segmentation BiSeNet node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
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
                # Convert OpenDR heatmap image to ROS2 image message using bridge and publish it
                self.heatmap_publisher.publish(self.bridge.to_ros_image(Image(heatmap_o), encoding='bgr8'))
        except Exception:
            rospy.logwarn('Failed to generate prediction.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=str, default="/opendr/heatmap")
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
                               output_rgb_image_topic=args.output_rgb_image_topic)
    bisenet_node.listen()


if __name__ == '__main__':
    main()
