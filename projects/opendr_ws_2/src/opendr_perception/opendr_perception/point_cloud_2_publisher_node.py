#!/usr/bin/env python3.6
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
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 as ROS_PointCloud2
from opendr_ros2_bridge import ROS2Bridge

from opendr.perception.panoptic_segmentation import SemanticKittiDataset


class PointCloud2DatasetNode(Node):

    def __init__(self,
                 path: str = './datasets/semantickitti',
                 split: str = 'valid',
                 output_point_cloud_2_topic: str = "/opendr/dataset_point_cloud2"
                 ):
        super().__init__("opendr_point_cloud_2_dataset_node")

        """
        Creates a ROS Node for publishing a PointCloud2 message from a DatasetIterator
        :param dataset: DatasetIterator from which we are reading the point cloud
        :type dataset: DatasetIterator
        :param output_point_cloud_2_topic: Topic to which we are publishing the point cloud
        :type output_point_cloud_2_topic: str
        """

        self.path = path
        self.split = split
        self._ros2_bridge = ROS2Bridge()

        if output_point_cloud_2_topic is not None:
            self.output_point_cloud_2_publisher = self.create_publisher(
                output_point_cloud_2_topic, ROS_PointCloud2, queue_size=10
            )

    def start(self):
        """
        Starts the ROS Node
        """
        if self._init_dataset():
            self.get_logger().info("Starting point cloud 2 dataset node")
        i = 0 
        print("Starting point cloud 2 publisher")
        while rclpy.ok():
            print("Publishing point cloud 2 message")
            point_cloud = self.dataset[i % len(self.dataset)][0]
            self.get_logger().info("Publishing point_cloud_2 [" + str(i) + "]")
            message = self._ros2_bridge.to_ros_point_cloud2(point_cloud, 
                                                            self.get_clock().now().to_msg(),
                                                            ROS_PointCloud2)
            self.point_cloud_2_publisher.publish(message)
            i += 1

            time.sleep(0.1)

    def _init_dataset(self):
        try:
            self.dataset = SemanticKittiDataset(path=self.path, split=self.split)
            return True
        except FileNotFoundError:
            self.get_logger().error("Dataset not found. Please download the dataset and extract it or enter available path.")
            return False

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset_path', type=str, default='./datasets/semantickitti',
                        help='listen to pointclouds on this topic')
    parser.add_argument('-s', '--split', type=str, default='valid',
                        help='split of the dataset to use (train, valid, test)')
    parser.add_argument('-o', '--output_point_cloud_2_topic', type=str, default='/opendr/dataset_point_cloud2',
                        help='topic for the output point cloud')
    args = parser.parse_args()

    dataset_node = PointCloud2DatasetNode(args.dataset_path, args.split, args.output_point_cloud_2_topic)

    dataset_node.start()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dataset_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
