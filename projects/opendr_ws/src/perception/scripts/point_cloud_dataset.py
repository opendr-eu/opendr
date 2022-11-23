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

import os
import rospy
import time
from sensor_msgs.msg import PointCloud as ROS_PointCloud
from opendr_bridge import ROSBridge
from opendr.engine.datasets import DatasetIterator
from opendr.perception.object_detection_3d import KittiDataset, LabeledPointCloudsDatasetIterator


class PointCloudDatasetNode:
    def __init__(
        self,
        dataset: DatasetIterator,
        output_point_cloud_topic="/opendr/dataset_point_cloud",
    ):
        """
        Creates a ROS Node for publishing dataset point clouds
        """

        # Initialize the face detector
        self.dataset = dataset
        # Initialize OpenDR ROSBridge object
        self.bridge = ROSBridge()

        if output_point_cloud_topic is not None:
            self.output_point_cloud_publisher = rospy.Publisher(
                output_point_cloud_topic, ROS_PointCloud, queue_size=10
            )

    def start(self):
        i = 0

        while not rospy.is_shutdown():

            point_cloud = self.dataset[i % len(self.dataset)][0]  # Dataset should have a (PointCloud, Target) pair as elements

            rospy.loginfo("Publishing point_cloud [" + str(i) + "]")
            message = self.bridge.to_ros_point_cloud(
                point_cloud
            )
            self.output_point_cloud_publisher.publish(message)

            time.sleep(0.1)

            i += 1


if __name__ == "__main__":

    rospy.init_node('opendr_point_cloud_dataset')

    dataset_path = "KITTI/opendr_nano_kitti"

    if not os.path.exists(dataset_path):
        dataset_path = KittiDataset.download_nano_kitti(
            "KITTI", kitti_subsets_path="../../src/opendr/perception/object_detection_3d/datasets/nano_kitti_subsets",
            create_dir=True,
        ).path

    dataset = LabeledPointCloudsDatasetIterator(
        dataset_path + "/training/velodyne_reduced",
        dataset_path + "/training/label_2",
        dataset_path + "/training/calib",
    )

    dataset_node = PointCloudDatasetNode(dataset)
    dataset_node.start()
