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

import sys
from pathlib import Path
import argparse
from typing import Optional

import rclpy
from rclpy.node import Node
import matplotlib
from sensor_msgs.msg import PointCloud2 as ROS_PointCloud2

from opendr_ros2_bridge import ROS2Bridge
from opendr.perception.panoptic_segmentation import EfficientLpsLearner

# Avoid having a matplotlib GUI in a separate thread in the visualize() function
matplotlib.use('Agg')


class EfficientLpsNode(Node):

    def __init__(self,
                 input_pcl_topic: str,
                 checkpoint: str,
                 output_rgb_visualization_topic: Optional[str] = None
                 ):
        """
        Initialize the EfficientLPS ROS node and create an instance of the respective learner class.
        :param input_pcl_topic: The name of the input point cloud topic.
        :type input_pcl_topic: str
        :param checkpoint: The path to the checkpoint file or the name of the pre-trained model.
        :type checkpoint: str
        :param output_rgb_visualization_topic: The name of the output RGB visualization topic.
        :type output_rgb_visualization_topic: str
        """
        super().__init__('opendr_efficient_lps_node')

        self.input_pcl_topic = input_pcl_topic
        self.checkpoint = checkpoint
        self.output_rgb_visualization_topic = output_rgb_visualization_topic

        # Initialize all ROS2 related things
        self._bridge = ROS2Bridge()
        self._visualization_publisher = None

        # Initialize the panoptic segmentation network
        config_file = Path(sys.modules[
            EfficientLpsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_semantickitti.py'
        self._learner = EfficientLpsLearner(str(config_file))

        # Other
        self._tmp_folder = Path(__file__).parent.parent / 'tmp' / 'efficientlps'
        self._tmp_folder.mkdir(exist_ok=True, parents=True)

    def _init_learner(self) -> bool:
        """
        The model can be initialized via
        1. downloading pre-trained weights for SemanticKITTI.
        2. passing a path to an existing checkpoint file.
        This has not been done in the __init__() function since logging is available only once the node is registered.
        """
        if self.checkpoint in ['semantickitti']:
            file_path = EfficientLpsLearner.download(str(self._tmp_folder),
                                                     trained_on=self.checkpoint)
            self.checkpoint = file_path

        if self._learner.load(self.checkpoint):
            self.get_logger().info('Successfully loaded the checkpoint.')
            return True
        else:
            self.get_logger().error('Failed to load the checkpoint.')
            return False

    def _init_publisher(self):
        """
        Set up the publishers as requested by the user.
        """
        if self.output_rgb_visualization_topic is not None:
            self._visualization_publisher = self.create_publisher(ROS_PointCloud2,
                                                                  self.output_rgb_visualization_topic,
                                                                  10)

    def _init_subscriber(self):
        """
        Subscribe to all relevant topics.
        """
        self.pointcloud2_subscriber = self.create_subscription(ROS_PointCloud2,
                                                               self.input_pcl_topic,
                                                               self.callback,
                                                               1)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        if self._init_learner():
            self._init_publisher()
            self._init_subscriber()
            self.get_logger().info('EfficientLPS node started!')
            rclpy.spin(self)

            # Destroy the node explicitly
            # (optional - otherwise it will be done automatically
            # when the garbage collector destroys the node object)
            self.destroy_node()
            rclpy.shutdown()

    def callback(self, data: ROS_PointCloud2):
        """
        Predict the panoptic segmentation map from the input point cloud and publish the results.
        :param data: PointCloud2 data message
        :type data: sensor_msgs.msg.PointCloud2
        """

        pointcloud = self._bridge.from_ros_point_cloud2(data)

        try:
            prediction = self._learner.infer(pointcloud)

        except Exception as e:
            self.get_logger().error('Failed to perform inference: {}'.format(e))
            return

        try:
            # The output topics are only published if there is at least one subscriber
            if self._visualization_publisher is not None and self._visualization_publisher.get_subscription_count() > 0:
                pointcloud_visualization = EfficientLpsLearner.visualize(pointcloud,
                                                                         prediction,
                                                                         return_pointcloud=True,
                                                                         return_pointcloud_type="panoptic")
                ros_pointcloud2_msg = self._bridge.to_ros_point_cloud2(pointcloud_visualization,
                                                                       self.get_clock().now().to_msg(),
                                                                       channels='rgb')
                self._visualization_publisher.publish(ros_pointcloud2_msg)

        except Exception as e:
            self.get_logger().error('Failed to publish the results: {}'.format(e))


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_point_cloud_2_topic', type=str, default='/opendr/dataset_point_cloud2',
                        help='Point Cloud 2 topic provided by either a \
                              point_cloud_2_publisher_node or any other 3D Point Cloud 2 Node')
    parser.add_argument('-c', '--checkpoint', type=str, default='semantickitti',
                        help='Download pretrained models [semantickitti] or load from the provided path')
    parser.add_argument('-o', '--output_rgb_visualization_topic', type=str, default="/opendr/panoptic",
                        help='Publish the rgb visualization on this topic')

    args = parser.parse_args()
    efficient_lps_node = EfficientLpsNode(args.input_point_cloud_2_topic,
                                          args.checkpoint,
                                          args.output_rgb_visualization_topic)
    efficient_lps_node.listen()


if __name__ == '__main__':
    main()
