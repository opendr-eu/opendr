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

import rospy
import time
import argparse

from sensor_msgs.msg import PointCloud2 as ROS_PointCloud2
from opendr_bridge import ROSBridge

from opendr.perception.panoptic_segmentation import SemanticKittiDataset
from opendr.perception.panoptic_segmentation import EfficientLpsLearner


class PointCloud2DatasetNode:
    def __init__(self,
                 path: str = './datasets/semantickitti',
                 split: str = 'valid',
                 output_point_cloud_2_topic: str = "/opendr/dataset_point_cloud2"
                 ):
        """
        Creates a ROS Node for publishing dataset point clouds
        :param path: path to the dataset
        :type path: str
        :param split: split of the dataset to use (train, valid, test)
        :type split: str
        :param output_point_cloud_2_topic: topic for the output point cloud
        :type output_point_cloud_2_topic: str
        """

        self.path = path
        self.split = split
        # Initialize OpenDR ROSBridge object
        self.bridge = ROSBridge()

        if output_point_cloud_2_topic is not None:
            self.output_point_cloud_2_publisher = rospy.Publisher(
                output_point_cloud_2_topic, ROS_PointCloud2, queue_size=10
            )

    def start(self):
        """
        Starts the node
        """
        if self._init_dataset():
            rospy.loginfo("Starting point_cloud_2 dataset node")
            rospy.init_node('opendr_point_cloud_2_dataset_node')

            i = 0
            while not rospy.is_shutdown():

                point_cloud = self.dataset[i % len(self.dataset)][0]

                rospy.loginfo("Publishing point_cloud_2 [" + str(i) + "]")
                message = self.bridge.to_ros_point_cloud2(
                    point_cloud
                )
                self.output_point_cloud_2_publisher.publish(message)

                time.sleep(0.1)

                i += 1

    def _init_dataset(self):
        try:
            self.dataset = SemanticKittiDataset(path=self.path, split=self.split)
            return True
        except FileNotFoundError:
            rospy.logerr("Dataset not found. Please download the dataset and extract it or enter available path.")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset_path', type=str, default='./datasets/semantickitti',
                        help='listen to pointclouds on this topic')
    parser.add_argument('-s', '--split', type=str, default='valid',
                        help='split of the dataset to use (train, valid, test)')
    parser.add_argument('-o', '--output_point_cloud_2_topic', type=str, default='/opendr/dataset_point_cloud2',
                        help='topic for the output point cloud')
    parser.add_argument('-t', '--test_data', action='store_true',
                        help='Use uploaded test data on the FTP server')
    args = parser.parse_args()

    if args.test_data:
        args.dataset_path = EfficientLpsLearner.download(args.dataset_path, mode="test_data", prepare_data=True)

    dataset_node = PointCloud2DatasetNode(args.dataset_path, args.split, args.output_point_cloud_2_topic)
    dataset_node.start()
