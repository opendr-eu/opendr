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
from typing import Optional, List

import matplotlib
import numpy as np
import rospy
from sensor_msgs.msg import Image as ROSImage, PointCloud as ROSPointCloud

from opendr_bridge import ROSBridge
from opendr.engine.data import PointCloud
from opendr.perception.panoptic_segmentation import EfficientLpsLearner, EfficientPsLearner

# Avoid having a matplotlib GUI in a separate thread in the visualize() function
matplotlib.use("Agg")


def join_arrays(arrays: List[np.ndarray]):
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


class EfficientLpsNode:
    def __init__(self,
                 checkpoint: str,
                 input_pcl_topic: str,
                 output_topic: Optional[str] = None,
                 output_visualization_topic: Optional[str] = None,
                 projected_output: bool = False,
                 detailed_visualization: bool = False
                 ):
        """
        Initialize the EfficientLPS ROS node and create an instance of the respective learner class.
        :param checkpoint: Path to a saved model
        :type checkpoint: str
        :param input_pcl_topic: ROS topic for the input image stream
        :type input_pcl_topic: str
        :param output_topic: ROS topic for the predicted semantic and instance maps
        :type output_topic: str
        :param output_visualization_topic: ROS topic for the generated visualization of the panoptic map
        :type output_visualization_topic: str
        :param projected_output: Publish predictions as a 2D Projection.
        :type projected_output: bool
        :param detailed_visualization: Visualization will be published in detail as separate maps in a single image.
        :type detailed_visualization: bool
        """
        
        self.checkpoint = checkpoint
        self.input_pcl_topic = input_pcl_topic
        self.output_topic = output_topic
        self.output_visualization_topic = output_visualization_topic
        self.projected_output = projected_output
        self.detailed_visualization = detailed_visualization

        # Initialize all ROS related things
        self._bridge = ROSBridge()
        self._instance_publisher = None
        self._semantic_publisher = None
        self._visualization_publisher = None

        # Initialize the panoptic segmentation network
        self._learner = EfficientLpsLearner()

    def _init_learner(self) -> bool:
        """
        Load the weights from the specified checkpoint file.

        This has not been done in the __init__() function since logging is available only once the node is registered.
        """
        
        status = self._learner.load(self.checkpoint)
        if status:
            rospy.loginfo("Successfully loaded the checkpoint.")
        else:
            rospy.logerr("Failed to load the checkpoint.")
        
        return status

    def _init_subscribers(self):
        """
        Initialize the Subscribers to all relevant topics.
        """
        
        rospy.Subscriber(self.input_pcl_topic, ROSPointCloud, self.callback)

    def _init_publisher(self):
        """
        Initialize the Publishers as requested by the user.
        """
        
        if self.output_topic is not None:
            if self.projected_output:
                self._instance_publisher = rospy.Publisher(f"{self.output_topic}/instance", ROSImage,
                                                           queue_size=10)
                self._semantic_publisher = rospy.Publisher(f"{self.output_topic}/semantic", ROSImage,
                                                           queue_size=10)
            else:
                self._instance_publisher = rospy.Publisher(self.output_topic, ROSPointCloud,
                                                           queue_size=10)
                self._semantic_publisher = None
                
        if self.output_visualization_topic is not None:
            self._visualization_publisher = rospy.Publisher(self.output_visualization_topic, ROSImage, queue_size=10)

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

    def callback(self, data: ROSPointCloud):
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
                    panoptic_image = EfficientPsLearner.visualize(prediction[2], prediction[:2], show_figure=False,
                                                                  detailed=self.detailed_visualization)
                else:
                    panoptic_image = EfficientLpsLearner.visualize(pointcloud, prediction[:2], show_figure=False)
                
                self._visualization_publisher.publish(self._bridge.to_ros_image(panoptic_image))

            if self._instance_publisher is not None and \
                    self._instance_publisher.get_num_connections() > 0:

                if self.projected_output:
                    self._instance_publisher.publish(self._bridge.to_ros_image(prediction[0]))
                else:
                    labeled_pc = PointCloud(join_arrays([pointcloud.data, prediction[0], prediction[1]]))
                    self._instance_publisher.publish(self._bridge.to_ros_point_cloud(labeled_pc))

            if self._semantic_publisher is not None and \
                    self._semantic_publisher.get_num_connections() > 0 and \
                    self.projected_output:
                self._semantic_publisher.publish(self._bridge.to_ros_image(prediction[1]))

        except Exception:
            rospy.logwarn("Failed to generate prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str,
                        help="Path to the model weights to be loaded.")
    parser.add_argument("pcl_topic", type=str,
                        help="ROS Topic to listen to point clouds from.")
    parser.add_argument("--output_topic", type=str,
                        help="ROS Topic where to publish the semantic and instance predictions.")
    parser.add_argument("--visualization_topic", type=str,
                        help="Publish the panoptic segmentation map as an RGB image on this topic or a more detailed \
                              overview if using the --detailed_visualization flag")
    parser.add_argument("--projected_output", action="store_true",
                        help="Compute the predictions and visualizations as 2D projected maps if True, \
                        otherwise as additional channels in a Point Cloud.")
    parser.add_argument("--detailed_visualization", action="store_true",
                        help="If projected_output is set to True, generate a combined overview of the input RGB image \
                             and the semantic, instance, and panoptic segmentation maps. Otherwise useless.")
    args = parser.parse_args()

    efficient_ps_node = EfficientLpsNode(args.checkpoint, args.image_topic,
                                         output_topic=args.output_topic,
                                         output_visualization_topic=args.visualization_topic,
                                         projected_output=args.projected_output,
                                         detailed_visualization=args.detailed_visualization)
    efficient_ps_node.listen()
