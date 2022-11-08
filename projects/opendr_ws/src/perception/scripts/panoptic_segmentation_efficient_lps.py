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

import sys
from pathlib import Path
import argparse
from typing import Optional, List

import numpy as np
import matplotlib
import rospy
from sensor_msgs.msg import PointCloud2 as ROS_PointCloud2
from sensor_msgs.msg import Image as ROS_Image

from opendr_bridge import ROSBridge
from opendr.perception.panoptic_segmentation import EfficientLpsLearner

# Avoid having a matplotlib GUI in a separate thread in the visualize() function
matplotlib.use("Agg")


class EfficientLpsNode:

    def __init__(self,
                 input_rgb_pcl_topic: str,
                 checkpoint: str,
                 output_heatmap_topic: Optional[str] = None,
                 output_rgb_image_topic: Optional[str] = None,
                 projected_output: bool = False,
                 ):
        """
        Initialize the EfficientLPS ROS node and create an instance of the respective learner class.
        :param input_rgb_pcl_topic: ROS topic for the input point cloud
        :type input_rgb_pcl_topic: str
        :param checkpoint: This is either a path to a saved model or SemanticKITTI to download
            pre-trained model weights.
        :type checkpoint: str
        :param output_heatmap_topic: ROS topic for the predicted semantic and instance maps
        :type output_topic: str
        :param output_visualization_topic: ROS topic for the generated visualization of the panoptic map
        :type output_visualization_topic: str
        :param projected_output: Publish predictions as a 2D Projection.
        :type projected_output: bool
        """

        self.checkpoint = checkpoint
        self.input_rgb_pcl_topic = input_rgb_pcl_topic
        self.output_heatmap_pointcloud_topic = output_heatmap_topic
        self.output_rgb_visualization_topic = output_rgb_image_topic
        self.projected_output = projected_output

        # Initialize all ROS related things
        self._bridge = ROSBridge()
        self._instance_heatmap_publisher = None
        self._semantic_heatmap_publisher = None
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
            rospy.loginfo('Successfully loaded the checkpoint.')
            return True
        else:
            rospy.logerr('Failed to load the checkpoint.')
            return False

    def _init_subscribers(self):
        """
        Initialize the Subscribers to all relevant topics.
        """

        rospy.Subscriber(self.input_rgb_pcl_topic, ROS_PointCloud2, self.callback)

    def _init_publisher(self):
        """
        Set up the publishers as requested by the user.
        """
        if self.output_heatmap_pointcloud_topic is not None:
            if self.projected_output:
                self._instance_heatmap_publisher = rospy.Publisher(
                    f"{self.output_heatmap_pointcloud_topic}/instance", ROS_Image, queue_size=10)
                self._semantic_heatmap_publisher = rospy.Publisher(
                    f"{self.output_heatmap_pointcloud_topic}/semantic", ROS_Image, queue_size=10)
            else:
                self._instance_heatmap_publisher = rospy.Publisher(
                    self.output_heatmap_pointcloud_topic, ROS_PointCloud2, queue_size=10)
                self._semantic_heatmap_publisher = None
        if self.output_rgb_visualization_topic is not None:
            self._visualization_publisher = rospy.Publisher(self.output_rgb_visualization_topic,
                                                            ROS_Image, queue_size=10)

    def _join_arrays(self, arrays: List[np.ndarray]):
        """
        Function for efficiently concatenating numpy arrays.

        :param arrays: List of numpy arrays to be concatenated
        :type arrays: List[np.ndarray]

        :return: Array comprised of the concatenated inputs.
        :rtype: np.ndarray
        """

        sizes = np.array([a.itemsize for a in arrays])
        offsets = np.r_[0, sizes.cumsum()]
        n = len(arrays[0])
        joint = np.empty((n, offsets[-1]), dtype=np.uint8)

        for a, size, offset in zip(arrays, sizes, offsets):
            joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)

        dtype = sum((a.dtype.descr for a in arrays), [])

        return joint.ravel().view(dtype)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input point clouds without being in a trained state.
        """

        rospy.init_node("efficient_lps", anonymous=True)
        rospy.loginfo("EfficientLPS node started!")
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
        pointcloud = self._bridge.from_ros_point_cloud(data)

        try:
            # Get a list of two OpenDR heatmaps: [instance map, semantic map, depth map] if projected_output is True,
            # or a list of numpy arrays: [instance labels, semantic labels] otherwise.
            prediction = self._learner.infer(pointcloud, projected=self.projected_output)

            # The output topics are only published if there is at least one subscriber
            if self._visualization_publisher is not None and \
                    self._visualization_publisher.get_num_connections() > 0:
                if self.projected_output:
                    projected_output = prediction[2]
                    panoptic_image = EfficientLpsLearner.visualize(projected_output, prediction[:2], show_figure=False)
                else:
                    panoptic_image = EfficientLpsLearner.visualize(pointcloud, prediction[:2], show_figure=False)

                self._visualization_publisher.publish(self._bridge.to_ros_image(panoptic_image))
            if self._instance_heatmap_publisher is not None and \
                    self._instance_heatmap_publisher.get_num_connections() > 0:

                if self.projected_output:
                    self._instance_heatmap_publisher.publish(self._bridge.to_ros_image(prediction[0]))
                else:
                    labeled_pc = ROS_PointCloud2(self._join_arrays([pointcloud.data, prediction[0], prediction[1]]))
                    self._instance_heatmap_publisher.publish(self._bridge.to_ros_point_cloud(labeled_pc))

            if self._semantic_heatmap_publisher is not None and \
                    self._semantic_heatmap_publisher.get_num_connections() > 0:
                if self.projected_output:
                    self._semantic_heatmap_publisher.publish(self._bridge.to_ros_image(prediction[1]))
                else:
                    rospy.logwarn("Semantic heatmap cannot be published in non-projected mode.")

        except Exception as e:
            rospy.logwarn(f'Failed to generate prediction: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_rgb_pcl_topic', type=str, default='/usb_cam/pcl_raw',
                        help='listen to RGB pointclouds on this topic')
    parser.add_argument('--checkpoint', type=str, default='semantickitti',
                        help='download pretrained models [semantickitti] or load from the provided path')
    parser.add_argument('--output_heatmap_topic', type=str, default='/opendr/panoptic',
                        help='publish the semantic and instance maps or pointcloud on this topic \
                            as "OUTPUT_HEATMAP_TOPIC/semantic" and "OUTPUT_HEATMAP_TOPIC/instance"')
    parser.add_argument('--output_rgb_visualization_topic', type=str,
                        default='/opendr/panoptic/rgb_visualization',
                        help='publish the panoptic segmentation map as an RGB image on this topic')
    parser.add_argument("--projected_output", action="store_true",
                        help="Compute the predictions and visualizations as 2D projected maps if True, \
                        otherwise as additional channels in a Point Cloud.")

    args = parser.parse_args()

    efficient_ps_node = EfficientLpsNode(args.input_rgb_pcl_topic,
                                         args.checkpoint,
                                         args.output_heatmap_topic,
                                         args.output_rgb_visualization_topic,
                                         args.projected_output)

    efficient_ps_node.listen()
