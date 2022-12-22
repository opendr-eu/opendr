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
import os
import rospy
import time
from sensor_msgs.msg import Image as ROS_Image, ChannelFloat32 as ROS_ChannelFloat32
from opendr_bridge import ROSBridge
from opendr.engine.datasets import DatasetIterator
from opendr.perception.continual_slam.datasets.kitti import KittiDataset


class ContinualSlamDatasetNode:

    def __init__(self, 
                 Dataset: DatasetIterator,
                 output_image_topic: str = "/opendr/dataset_image",
                 output_velocity_topic: str = "opendr/velocity",
                 output_distance_topic: str = "opendr/distance",
                 dataset_fps: int = 10):
        """
        Creates a ROS Node for publishing dataset images
        :param Dataset: DatasetIterator object
        :type Dataset: DatasetIterator
        :param output_image_topic: ROS topic to publish images
        :type output_image_topic: str
        :param output_velocity_topic: ROS topic to publish velocities
        :type output_velocity_topic: str
        :param output_distance_topic: ROS topic to publish distances
        :type output_distance_topic: str
        :param dataset_fps: Dataset frame rate
        :type dataset_fps: int
        """

        self.dataset = Dataset
        self.bridge = ROSBridge()
        self.delay = 1.0 / dataset_fps

        self.output_image_topic = output_image_topic
        self.output_velocity_topic = output_velocity_topic
        self.output_distance_topic = output_distance_topic

    def _init_publisher(self):

        self.output_image_publisher = rospy.Publisher(
            self.output_image_topic, ROS_Image, queue_size=10)
        self.output_velocity_publisher = rospy.Publisher(
            self.output_velocity_topic, ROS_ChannelFloat32, queue_size=10)
        self.output_distance_publisher = rospy.Publisher(
            self.output_distance_topic, ROS_ChannelFloat32, queue_size=10)

    def _publish(self):

        rospy.loginfo("Start publishing dataset images")
        i = 0
        while not rospy.is_shutdown():
            data = self.dataset[i % len(self.dataset)][0] 
            # data is in format of {"image_id" : (image, velocity, distance)}
            # Get the image_id 
            image_id = list(data.keys())[0]
            # Get the image, velocity and distance
            image, velocity, distance = data[image_id]
            # Convert image to ROS Image
            message = self.bridge.to_ros_image(image)
            # Publish the image
            self.output_image_publisher.publish(message)
            # Convert velocity to ROS ChannelFloat32
            message = self.bridge.to_ros_channel_float32(image_id, [velocity])
            # Publish the velocity
            self.output_velocity_publisher.publish(message)
            # Convert distance to ROS ChannelFloat32
            message = self.bridge.to_ros_channel_float32(image_id, [distance])
            # Publish the distance
            self.output_distance_publisher.publish(message)

            rospy.loginfo("Published image {}".format(image_id))
            i += 1
            time.sleep(self.delay)

    def run(self):
            
            rospy.init_node("continual_slam_dataset_node", anonymous=True)
            self._init_publisher()
            self._publish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/canakcia/Desktop/",
                        help="Path to the dataset")
    parser.add_argument("--output_image_topic", type=str, default="/opendr/dataset_image",
                        help="ROS topic to publish images")
    parser.add_argument("--output_velocity_topic", type=str, default="/opendr/velocity",    
                        help="ROS topic to publish velocities")
    parser.add_argument("--output_distance_topic", type=str, default="/opendr/distance",
                        help="ROS topic to publish distances")
    parser.add_argument("--dataset_fps", type=int, default=120,
                        help="Dataset frame rate")
    args = parser.parse_args()

    dataset = KittiDataset(args.dataset_path)
    node = ContinualSlamDatasetNode(dataset, args.output_image_topic, args.output_velocity_topic, args.output_distance_topic, args.dataset_fps)
    node.run()

if __name__ == "__main__":
    main()




