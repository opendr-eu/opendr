#!/usr/bin/env python
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

import argparse
import torch
import rospy
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.engine.data import Image
from opendr.perception.semantic_segmentation.bisenet.bisenet_learner import BisenetLearner
import numpy as np
import cv2


class BisenetNode:
    def __init__(self,
                 input_image_topic: str = "/image_publisher/raw_image",
                 output_heatmap_topic: str = None,
                 device="cuda"
                 ):
        """
        Initialize the Bisenet ROS node and create an instance of the respective learner class.
        :param input_image_topic: ROS topic for the input image stream
        :type input_image_topic: str
        :param output_heatmap_topic: ROS topic for the predicted heatmap
        :type output_heatmap_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        self.input_image_topic = input_image_topic
        self.output_heatmap_topic = output_heatmap_topic

        if self.output_heatmap_topic is not None:
            self._heatmap_publisher = rospy.Publisher(f'{self.output_heatmap_topic}/semantic', ROS_Image, queue_size=10)
        else:
            self._heatmap_publisher = None

        rospy.Subscriber(self.input_image_topic, ROS_Image, self.callback)

        # Initialize OpenDR ROSBridge object
        self._bridge = ROSBridge()

        # Initialize the semantic segmentation model
        self._learner = BisenetLearner(device=device)
        self._learner.download(path="bisenet_camvid")
        self._learner.load("bisenet_camvid")

        self._colors = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('bisenet', anonymous=True)
        rospy.loginfo("Bisenet node started!")
        rospy.spin()

    def callback(self, data: ROS_Image):
        """
        Predict the heatmap from the input image and publish the results.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image to OpenDR Image
        image = self._bridge.from_ros_image(data)

        try:
            # Retrieve the OpenDR heatmap
            prediction = self._learner.infer(image)

            if self._heatmap_publisher is not None and self._heatmap_publisher.get_num_connections() > 0:
                heatmap_np = prediction.numpy()
                heatmap_o = self._colors[heatmap_np]
                heatmap_o = cv2.resize(np.uint8(heatmap_o), (960, 720))
                self._heatmap_publisher.publish(self._bridge.to_ros_image(Image(heatmap_o), encoding='bgr8'))

        except Exception:
            rospy.logwarn('Failed to generate prediction.')


if __name__ == '__main__':
    # Select the device for running the
    try:
        if torch.cuda.is_available():
            print("GPU found.")
            device = "cuda"
        else:
            print("GPU not found. Using CPU instead.")
            device = "cpu"
    except:
        device = "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('image_topic', type=str, help='listen to images on this topic')
    parser.add_argument('--heatmap_topic', type=str, help='publish the heatmap on this topic')
    args = parser.parse_args()

    bisenet_node = BisenetNode(device=device, input_image_topic=args.image_topic, output_heatmap_topic=args.heatmap_topic)
    bisenet_node.listen()
