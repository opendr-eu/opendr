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

import argparse
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROS2Bridge
from opendr.engine.datasets import DatasetIterator
from opendr.perception.object_tracking_2d import MotDataset, RawMotDatasetIterator


class ImageDatasetNode(Node):
    def __init__(
            self,
            dataset: DatasetIterator,
            output_rgb_image_topic="/opendr/dataset_image",
            data_fps=10,
    ):
        """
        Creates a ROS2 Node for publishing dataset images
        """

        super().__init__('opendr_image_dataset_node')

        self.dataset = dataset
        self.bridge = ROS2Bridge()
        self.timer = self.create_timer(1.0 / data_fps, self.timer_callback)
        self.sample_index = 0

        self.output_image_publisher = self.create_publisher(
            ROS_Image, output_rgb_image_topic, 1
        )
        self.get_logger().info("Publishing images.")

    def timer_callback(self):
        image = self.dataset[self.sample_index % len(self.dataset)][0]
        # Dataset should have an (Image, Target) pair as elements

        message = self.bridge.to_ros_image(
            image, encoding="bgr8"
        )
        self.output_image_publisher.publish(message)

        self.sample_index += 1


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", help="Path to a dataset",
                        type=str, default="MOT")
    parser.add_argument(
        "-ks", "--mot20_subsets_path", help="Path to mot20 subsets",
        type=str, default=os.path.join(
            "..", "..", "src", "opendr", "perception", "object_tracking_2d",
            "datasets", "splits", "nano_mot20.train"
        )
    )
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name to publish the data",
                        type=str, default="/opendr/dataset_image")
    parser.add_argument("-f", "--fps", help="Data FPS",
                        type=float, default=30)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    mot20_subsets_path = args.mot20_subsets_path
    output_rgb_image_topic = args.output_rgb_image_topic
    data_fps = args.fps

    if not os.path.exists(dataset_path):
        dataset_path = MotDataset.download_nano_mot20(
            "MOT", True
        ).path

    dataset = RawMotDatasetIterator(
        dataset_path,
        {
            "mot20": mot20_subsets_path
        },
        scan_labels=False
    )
    dataset_node = ImageDatasetNode(
        dataset,
        output_rgb_image_topic=output_rgb_image_topic,
        data_fps=data_fps,
    )

    rclpy.spin(dataset_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dataset_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
