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

import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from ros2_bridge.bridge import ROS2Bridge


class SubscriberTesterNode(Node):

    def __init__(self):
        super().__init__('subscriber_tester_node')

        self.bridge = ROS2Bridge()

        self.sub_pose = self.create_subscription(Detection2DArray, "/opendr/poses", self.test_pose_callback, 10)

        self.sub_image = self.create_subscription(Image, "/opendr/image_pose_annotated", self.test_image_callback, 10)

    def test_pose_callback(self, msg):
        # Uncomment to print Pose message received
        # self.get_logger().info(f"Pose: {msg}")
        pass

    def test_image_callback(self, msg):
        # self.get_logger().info(f"Image: {msg}")
        # Show image after pose estimation
        if msg is not None:
            cv2.imshow("image", self.bridge.from_ros_image(msg).opencv())
            cv2.waitKey(5)


def main(args=None):
    rclpy.init(args=args)

    subscriber_tester = SubscriberTesterNode()

    try:
        rclpy.spin(subscriber_tester)
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber_tester.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
