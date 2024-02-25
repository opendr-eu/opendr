#!/usr/bin/env python3
# Copyright 2020-2024 OpenDR European Project
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
from numpy import mean

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class PerformanceNode(Node):

    def __init__(self, input_performance_topic="/opendr/performance", window_length=20):
        """
        Creates a ROS2 Node for subscribing to a node's performance topic to measure and log its performance in frames
        per second (FPS), rolling window.
        :param input_performance_topic: Topic from which we are reading the performance message
        :type input_performance_topic: str
        :param window_length: The length of the rolling average window
        :type window_length: int
        """
        super().__init__('opendr_performance_node')
        self.performance_subscriber = self.create_subscription(Float32, input_performance_topic, self.callback, 1)

        self.fps_window = []
        self.window_length = window_length
        self.get_logger().info("Performance node initialized.")

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        fps = data.data
        self.get_logger().info(f"Time per inference: {str(round(1.0 / fps, 4))} sec")
        while len(self.fps_window) < self.window_length:
            self.fps_window.append(fps)
        self.fps_window = self.fps_window[1:]
        self.fps_window.append(fps)
        self.get_logger().info(f"Average inferences per second: {round(mean(self.fps_window), 2)}")  # NOQA


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_performance_topic", help="Topic name for performance message",
                        type=str, default="/opendr/performance")
    parser.add_argument("-w", "--window", help="The window to use in frames to calculate running average FPS",
                        type=int, default=20)
    args = parser.parse_args()

    performance_node = PerformanceNode(input_performance_topic=args.input_performance_topic,
                                       window_length=args.window)

    rclpy.spin(performance_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    performance_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
