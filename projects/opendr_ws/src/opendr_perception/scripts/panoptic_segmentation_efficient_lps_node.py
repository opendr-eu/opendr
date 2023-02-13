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

import sys
from pathlib import Path
import argparse
from typing import Optional

import matplotlib
import rospy
from sensor_msgs.msg import PointCloud2 as ROS_PointCloud2

from opendr_bridge import ROSBridge
from opendr.perception.panoptic_segmentation import EfficientLpsLearner

# Avoid having a matplotlib GUI in a separate thread in the visualize() function
matplotlib.use("Agg")


class EfficientLpsNode:

    def __init__(self,
                 input_pcl_topic: str,
                 checkpoint: str,
                 output_heatmap_pointcloud_topic: Optional[str] = None,
                 ):
        """
        Initialize the EfficientLPS ROS node and create an instance of the respective learner class.
        :param input_pcl_topic: topic for the input point cloud
        :type input_pcl_topic: str
        :param checkpoint: This is either a path to a saved model or SemanticKITTI to download
            pre-trained model weights.
        :type checkpoint: str
        :param output_heatmap_pointcloud_topic: topic for the output 3D heatmap point cloud
        :type output_heatmap_pointcloud_topic: Optional[str]
        """

        self.checkpoint = checkpoint
        self.input_pcl_topic = input_pcl_topic
        self.output_heatmap_pointcloud_topic = output_heatmap_pointcloud_topic

        # Initialize all ROS related things
        self._bridge = ROSBridge()
        self._visualization_pointcloud_publisher = None

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
            rospy.loginfo('Successfully loaded the checkpoint.')
            return True
        else:
            rospy.logerr('Failed to load the checkpoint.')
            return False

    def _init_subscribers(self):
        """
        Initialize the Subscribers to all relevant topics.
        """

        rospy.Subscriber(self.input_pcl_topic, ROS_PointCloud2, self.callback)

    def _init_publisher(self):
        """
        Set up the publishers as requested by the user.
        """

        if self.output_heatmap_pointcloud_topic is not None:
            self._visualization_pointcloud_publisher = rospy.Publisher(self.output_heatmap_pointcloud_topic,
                                                                       ROS_PointCloud2, queue_size=10)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input point clouds without being in a trained state.
        """

        rospy.init_node("opendr_efficient_lps_node", anonymous=True)
        rospy.loginfo("EfficientLPS node started.")
        if self._init_learner():
            self._init_publisher()
            self._init_subscribers()
            rospy.spin()

    def callback(self, data: ROS_PointCloud2):
        """
        Predict the panoptic segmentation map from the input point cloud and publish the results.

        :param data: PointCloud data message
        :type data: sensor_msgs.msg.PointCloud
        """
        # Convert sensor_msgs.msg.Image to OpenDR Image
        pointcloud = self._bridge.from_ros_point_cloud2(data)

        try:
            # Get a list of two OpenDR heatmaps: [instance map, semantic map, depth map] if projected_output is True,
            # or a list of numpy arrays: [instance labels, semantic labels] otherwise.
            prediction = self._learner.infer(pointcloud)

            # The output topics are only published if there is at least one subscriber
            if self._visualization_pointcloud_publisher is not None and \
                    self._visualization_pointcloud_publisher.get_num_connections() > 0:
                # Get the RGB visualization of the panoptic map as pointcloud in OpenDR format
                pointcloud_visualization = EfficientLpsLearner.visualize(
                    pointcloud, prediction, return_pointcloud=True, return_pointcloud_type="panoptic")
                # Convert OpenDR Image to sensor_msgs.msg.PointCloud2
                ros_pointcloud2_msg = self._bridge.to_ros_point_cloud2(pointcloud_visualization, channels='rgb')
                # Publish the visualization
                self._visualization_pointcloud_publisher.publish(ros_pointcloud2_msg)

        except Exception as e:
            rospy.logwarn(f'Failed to generate prediction: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_point_cloud_2_topic', type=str, default='/opendr/dataset_point_cloud2',
                        help='Point Cloud 2 topic provided by either a \
                              point_cloud_2_publisher_node or any other 3D Point Cloud 2 Node')
    parser.add_argument('-c', '--checkpoint', type=str, default='semantickitti',
                        help='Download pretrained models [semantickitti] or load from the provided path')
    parser.add_argument('-o', '--output_heatmap_pointcloud_topic', type=str, default="/opendr/panoptic",
                        help='Publish the rgb visualization on this topic')

    args = parser.parse_args()

    efficient_lps_node = EfficientLpsNode(args.input_point_cloud_2_topic,
                                          args.checkpoint,
                                          args.output_heatmap_pointcloud_topic)

    efficient_lps_node.listen()
