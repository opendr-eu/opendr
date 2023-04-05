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
from numpy import mean

import rospy
from std_msgs.msg import Float32


class PerformanceNode:

    def __init__(self, input_performance_topic="/opendr/performance", window_length=20):
        """
        Creates a ROS Node for subscribing to a node's performance topic to measure and log its performance in frames
        per second (FPS), rolling window.
        :param input_performance_topic: Topic from which we are reading the performance message
        :type input_performance_topic: str
        :param window_length: The length of the rolling average window
        :type window_length: int
        """
        self.input_performance_topic = input_performance_topic
        self.fps_window = []
        self.window_length = window_length

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_performance_node', anonymous=True)
        rospy.Subscriber(self.input_performance_topic, Float32, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Performance node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        fps = data.data
        rospy.loginfo(f"Time per inference: {str(round(1.0 / fps, 4))} sec")
        while len(self.fps_window) < self.window_length:
            self.fps_window.append(fps)
        self.fps_window = self.fps_window[1:]
        self.fps_window.append(fps)
        rospy.loginfo(f"Average inferences per second: {round(mean(self.fps_window), 2)}")  # NOQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_performance_topic", help="Topic name for performance message",
                        type=str, default="/opendr/performance")
    parser.add_argument("-w", "--window", help="The window to use in frames to calculate running average FPS",
                        type=int, default=20)
    args = parser.parse_args()

    performance_node = PerformanceNode(input_performance_topic=args.input_performance_topic,
                                       window_length=args.window)
    performance_node.listen()


if __name__ == '__main__':
    main()
