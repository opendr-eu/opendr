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
        data_fps=10,
    ):
        """
        Creates a ROS Node for publishing dataset point clouds
        """

        self.dataset = dataset
        self.bridge = ROSBridge()
        self.delay = 1.0 / data_fps

        self.output_point_cloud_publisher = rospy.Publisher(
            output_point_cloud_topic, ROS_PointCloud, queue_size=10
        )

    def start(self):
        rospy.loginfo("Timing point cloud images")
        i = 0
        while not rospy.is_shutdown():
            point_cloud = self.dataset[i % len(self.dataset)][0]  # Dataset should have a (PointCloud, Target) pair as elements
            message = self.bridge.to_ros_point_cloud(
                point_cloud
            )
            self.output_point_cloud_publisher.publish(message)

            time.sleep(self.delay)
            i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path",
                        help="Path to a dataset. If does not exist, nano KITTI dataset will be downloaded there.",
                        type=str, default="KITTI/opendr_nano_kitti")
    parser.add_argument("-ks", "--kitti_subsets_path",
                        help="Path to kitti subsets. Used only if a KITTI dataset is downloaded",
                        type=str,
                        default="../../src/opendr/perception/object_detection_3d/datasets/nano_kitti_subsets")
    parser.add_argument("-o", "--output_point_cloud_topic", help="Topic name to publish the data",
                        type=str, default="/opendr/dataset_point_cloud")
    parser.add_argument("-f", "--fps", help="Data FPS",
                        type=float, default=10)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    kitti_subsets_path = args.kitti_subsets_path
    output_point_cloud_topic = args.output_point_cloud_topic
    data_fps = args.fps

    if not os.path.exists(dataset_path):
        dataset_path = KittiDataset.download_nano_kitti(
            "KITTI", kitti_subsets_path=kitti_subsets_path,
            create_dir=True,
        ).path

    dataset = LabeledPointCloudsDatasetIterator(
        dataset_path + "/training/velodyne_reduced",
        dataset_path + "/training/label_2",
        dataset_path + "/training/calib",
    )

    rospy.init_node('opendr_point_cloud_dataset_node', anonymous=True)

    dataset_node = PointCloudDatasetNode(
        dataset, output_point_cloud_topic=output_point_cloud_topic, data_fps=data_fps
    )

    dataset_node.start()
    rospy.loginfo("Point cloud dataset node started.")
    rospy.spin()


if __name__ == '__main__':
    main()
