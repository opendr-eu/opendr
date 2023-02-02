#!/usr/bin/env python3
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

import rospy
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge

from opendr.engine.data import Image
from opendr.perception.binary_high_resolution import BinaryHighResolutionLearner


class BinaryHighResolutionNode:

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw", output_heatmap_topic="/opendr/binary_hr_heatmap",
                 output_rgb_image_topic="/opendr/binary_hr_heatmap_visualization",
                 model_path=None, architecture="VGG_720p", device="cuda"):
        """
        Create a ROS Node for binary high resolution classification with Binary High Resolution.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_heatmap_topic: Topic to which we are publishing the heatmap in the form of a ROS image
        :type output_heatmap_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the heatmap image blended with the
        input image for visualization purposes
        :type output_rgb_image_topic: str
        :param model_path: The path to the directory of a trained model
        :type model_path: str
        :param architecture: Architecture used on trained model (`VGG_720p` or `VGG_1080p`)
        :type architecture: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        self.input_rgb_image_topic = input_rgb_image_topic

        if output_heatmap_topic is not None:
            self.heatmap_publisher = rospy.Publisher(output_heatmap_topic, ROS_Image, queue_size=1)
        else:
            self.heatmap_publisher = None

        if output_rgb_image_topic is not None:
            self.visualization_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=1)
        else:
            self.visualization_publisher = None

        self.bridge = ROSBridge()

        # Initialize the binary high resolution model
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
                                            "with opendr_ws as root or provide an absolute path.")
        rospy.loginfo(f"Model loaded from specified path: {model_path}")

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_binary_high_resolution_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Binary High Resolution node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        image = image.convert("channels_last")
        # Run learner to retrieve the OpenDR heatmap
        heatmap = self.learner.infer(image)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_heatmap_topic", help="Topic to which we are publishing the heatmap in the form "
                                                             "of a ROS image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/binary_hr_heatmap")
    parser.add_argument("-ov", "--output_rgb_image_topic", help="Topic to which we are publishing the heatmap image "
                                                                "blended with the input image for visualization purposes",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/binary_hr_heatmap_visualization")
    parser.add_argument("-m", "--model_path", help="Path to the directory of the trained model",
                        type=str, default="test_model")
    parser.add_argument("-a", "--architecture", help="Architecture used on trained model, either \"VGG_720p\" or \"VGG_1080p\","
                                                     "defaults to \"VGG_720p\"",
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
                                              model_path=args.model_path,
                                              architecture=args.architecture)
    binary_hr_node.listen()


if __name__ == '__main__':
    main()
