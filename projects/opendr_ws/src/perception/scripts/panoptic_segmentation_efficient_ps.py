#!/usr/bin/env python
# Copyright 2020-2021 OpenDR European Project
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
from typing import Optional

import matplotlib
import rospy
from sensor_msgs.msg import Image as ROS_Image

from opendr_bridge import ROSBridge
from opendr.perception.panoptic_segmentation import EfficientPsLearner

# Avoid having a matplotlib GUI in a separate thread in the visualize() function
matplotlib.use('Agg')


class EfficientPsNode:
    def __init__(self,
                 checkpoint: str,
                 input_image_topic: str,
                 output_heatmap_topic: Optional[str] = None,
                 output_visualization_topic: Optional[str] = None,
                 detailed_visualization: bool = False
                 ):
        """
        Initialize the EfficientPS ROS node and create an instance of the respective learner class.
        :param checkpoint: Path to a saved model
        :type checkpoint: str
        :param input_image_topic: ROS topic for the input image stream
        :type input_image_topic: str
        :param output_heatmap_topic: ROS topic for the predicted semantic and instance maps
        :type output_heatmap_topic: str
        :param output_visualization_topic: ROS topic for the generated visualization of the panoptic map
        :type output_visualization_topic: str
        """
        self.checkpoint = checkpoint
        self.input_image_topic = input_image_topic
        self.output_heatmap_topic = output_heatmap_topic
        self.output_visualization_topic = output_visualization_topic
        self.detailed_visualization = detailed_visualization

        # Initialize all ROS related things
        self._bridge = ROSBridge()
        self._instance_heatmap_publisher = None
        self._semantic_heatmap_publisher = None
        self._visualization_publisher = None

        # Initialize the panoptic segmentation network
        self._learner = EfficientPsLearner()

    def _init_learner(self) -> bool:
        """
        Load the weights from the specified checkpoint file.

        This has not been done in the __init__() function since logging is available only once the node is registered.
        """
        if self._learner.load(self.checkpoint):
            rospy.loginfo('Succesfully loaded the checkpoint.')
            return True
        else:
            rospy.logerr('Failed to load the checkpoint.')
            return False

    def _init_subscribers(self):
        """
        Subscribe to all relevant topics.
        """
        rospy.Subscriber(self.input_image_topic, ROS_Image, self.callback)

    def _init_publisher(self):
        """
        Set up the publishers as requested by the user.
        """
        if self.output_heatmap_topic is not None:
            self._instance_heatmap_publisher = rospy.Publisher(f'{self.output_heatmap_topic}/instance', ROS_Image,
                                                               queue_size=10)
            self._semantic_heatmap_publisher = rospy.Publisher(f'{self.output_heatmap_topic}/semantic', ROS_Image,
                                                               queue_size=10)
        if self.output_visualization_topic is not None:
            self._visualization_publisher = rospy.Publisher(self.output_visualization_topic, ROS_Image, queue_size=10)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        rospy.init_node('efficient_ps', anonymous=True)
        rospy.loginfo("EfficientPS node started!")
        if self._init_learner():
            self._init_publisher()
            self._init_subscribers()
            rospy.spin()

    def callback(self, data: ROS_Image):
        """
        Predict the panoptic segmentation map from the input image and publish the results.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image to OpenDR Image
        image = self._bridge.from_ros_image(data)

        try:
            # Retrieve a list of two OpenDR heatmaps: [instance map, semantic map]
            prediction = self._learner.infer(image)

            # The output topics are only published if there is at least one subscriber
            if self._visualization_publisher is not None and self._visualization_publisher.get_num_connections() > 0:
                panoptic_image = EfficientPsLearner.visualize(image, prediction, show_figure=False,
                                                              detailed=self.detailed_visualization)
                self._visualization_publisher.publish(self._bridge.to_ros_image(panoptic_image))

            if self._instance_heatmap_publisher is not None and self._instance_heatmap_publisher.get_num_connections() > 0:
                self._instance_heatmap_publisher.publish(self._bridge.to_ros_image(prediction[0]))
            if self._semantic_heatmap_publisher is not None and self._semantic_heatmap_publisher.get_num_connections() > 0:
                self._semantic_heatmap_publisher.publish(self._bridge.to_ros_image(prediction[1]))

        except Exception:
            rospy.logwarn('Failed to generate prediction.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='load the model weights from the provided path')
    parser.add_argument('image_topic', type=str, help='listen to images on this topic')
    parser.add_argument('--heatmap_topic', type=str, help='publish the semantic and instance maps on this topic')
    parser.add_argument('--visualization_topic', type=str,
                        help='publish the panoptic segmentation map as an RGB image on this topic or a more detailed \
                              overview if using the --detailed_visualization flag')
    parser.add_argument('--detailed_visualization', action='store_true',
                        help='generate a combined overview of the input RGB image and the semantic, instance, and \
                              panoptic segmentation maps')
    args = parser.parse_args()

    efficient_ps_node = EfficientPsNode(args.checkpoint,
                                        args.image_topic,
                                        args.heatmap_topic,
                                        args.visualization_topic,
                                        args.detailed_visualization)
    efficient_ps_node.listen()
