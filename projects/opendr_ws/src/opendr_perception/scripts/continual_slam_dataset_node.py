#!/usr/bin/env python
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

import os
import argparse
import time
from pathlib import Path

import rospy
from opendr_bridge import ROSBridge
from sensor_msgs.msg import Image as ROS_Image
from geometry_msgs.msg import Vector3Stamped as ROS_Vector3Stamped

from opendr.perception.continual_slam.datasets.kitti import KittiDataset


class ContinualSlamDatasetNode:

    def __init__(self,
                 dataset_path: str,
                 config_file_path: str,
                 output_image_topic: str = "/opendr/dataset/image",
                 output_distance_topic: str = "/opendr/imu/distance",
                 dataset_fps: float = 0.1):
        """
        Creates a ROS Node for publishing dataset images
        :param dataset_path: Path to the dataset
        :type dataset_path: str
        :param config_file_path: Path to the config file
        :type config_file_path: str
        :param output_image_topic: Topic to publish images to
        :type output_image_topic: str
        :param output_distance_topic: Topic to publish distance to
        :type output_distance_topic: str
        :param dataset_fps: Dataset FPS
        :type dataset_fps: float
        """
        self.dataset_path = dataset_path
        self.config_file_path = config_file_path
        self.bridge = ROSBridge()
        self.delay = 1.0 / dataset_fps

        self.output_image_topic = output_image_topic
        self.output_distance_topic = output_distance_topic

    def start(self):
        """
        Runs the node
        """
        rospy.init_node("continual_slam_dataset_node", anonymous=True)
        self._init_publisher()
        if self._init_dataset():
            self._publish()

    # Auxiliary functions
    def _init_publisher(self):
        """
        Initializes publishers
        """
        self.output_image_publisher = rospy.Publisher(self.output_image_topic,
                                                      ROS_Image,
                                                      queue_size=10)
        self.output_distance_publisher = rospy.Publisher(self.output_distance_topic,
                                                         ROS_Vector3Stamped,
                                                         queue_size=10)

    def _init_dataset(self):
        """
        Initializes dataset
        """
        env = os.getenv("OPENDR_HOME")
        config_file_path = os.path.join(env, self.config_file_path)
        if not Path(config_file_path).exists():
            raise FileNotFoundError("Config file not found")
        try:
            self.dataset = KittiDataset(self.dataset_path, config_file_path)
            return True
        except FileNotFoundError:
            rospy.logerr("Dataset path is incorrect. Please check the path")
            return False

    def _publish(self):
        """
        Publishes images and distances
        """
        rospy.loginfo("Start publishing dataset images")
        i = 0
        while not rospy.is_shutdown():
            if i == len(self.dataset)-2:
                break
            data = self.dataset[i][0]
            # Data is in format of {"image_id" : (image, velocity, distance)} for 3 past frames
            image_ids = list(data.keys())
            if len(data) < 3:
                i += 1
                continue
            image_t0, distance_t0 = data[image_ids[0]]

            stamp = rospy.Time.now()
            # Convert image to ROS Image
            image = self.bridge.to_ros_image(image=image_t0, frame_id=image_ids[0], time=stamp)
            # Convert velocity to ROS Vector3Stamped
            distance = self.bridge.to_ros_vector3_stamped(distance_t0, 0, 0, image_ids[0], stamp)
            # Publish the image and distance
            self.output_image_publisher.publish(image)
            self.output_distance_publisher.publish(distance)

            rospy.loginfo("Published image {}".format(image_ids[0]))
            rospy.loginfo("Published distance {}".format([distance_t0]))
            i += 1
            time.sleep(self.delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/canakcia/Desktop/kitti_dset/",
                        help="Path to the dataset")
    parser.add_argument("--config_file_path", type=str,
                        default="src/opendr/perception/continual_slam/configs/singlegpu_kitti.yaml",
                        help="Path to the config file")
    parser.add_argument("--output_image_topic", type=str, default="/cl_slam/image",
                        help="ROS topic to publish images")
    parser.add_argument("--output_distance_topic", type=str, default="/cl_slam/distance",
                        help="ROS topic to publish distances")
    parser.add_argument("--dataset_fps", type=float, default=3,
                        help="Dataset frame rate")
    args = parser.parse_args(rospy.myargv()[1:])

    node = ContinualSlamDatasetNode(args.dataset_path,
                                    args.config_file_path,
                                    args.output_image_topic,
                                    args.output_distance_topic,
                                    args.dataset_fps)
    node.start()


if __name__ == "__main__":
    main()
