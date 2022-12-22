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
                 input_rgb_image_topic: str,
                 checkpoint: str,
                 output_heatmap_topic: Optional[str] = None,
                 output_rgb_visualization_topic: Optional[str] = None,
                 detailed_visualization: bool = False
                 ):
        """
        Initialize the EfficientPS ROS node and create an instance of the respective learner class.
        :param checkpoint: This is either a path to a saved model or one of [cityscapes, kitti] to download
            pre-trained model weights.
        :type checkpoint: str
        :param input_rgb_image_topic: ROS topic for the input image stream
        :type input_rgb_image_topic: str
        :param output_heatmap_topic: ROS topic for the predicted semantic and instance maps
        :type output_heatmap_topic: str
        :param output_rgb_visualization_topic: ROS topic for the generated visualization of the panoptic map
        :type output_rgb_visualization_topic: str
        :param detailed_visualization: if True, generate a combined overview of the input RGB image and the
            semantic, instance, and panoptic segmentation maps and publish it on output_rgb_visualization_topic
        :type detailed_visualization: bool
        """
        self.input_rgb_image_topic = input_rgb_image_topic
        self.checkpoint = checkpoint
        self.output_heatmap_topic = output_heatmap_topic
        self.output_rgb_visualization_topic = output_rgb_visualization_topic
        self.detailed_visualization = detailed_visualization

        # Initialize all ROS related things
        self._bridge = ROSBridge()
        self._instance_heatmap_publisher = None
        self._semantic_heatmap_publisher = None
        self._visualization_publisher = None

        # Initialize the panoptic segmentation network
        config_file = Path(sys.modules[
                               EfficientPsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_cityscapes.py'
        self._learner = EfficientPsLearner(str(config_file))

        # Other
        self._tmp_folder = Path(__file__).parent.parent / 'tmp' / 'efficientps'
        self._tmp_folder.mkdir(exist_ok=True, parents=True)

    def _init_learner(self) -> bool:
        """
        The model can be initialized via
        1. downloading pre-trained weights for Cityscapes or KITTI.
        2. passing a path to an existing checkpoint file.

        This has not been done in the __init__() function since logging is available only once the node is registered.
        """
        if self.checkpoint in ['cityscapes', 'kitti']:
            file_path = EfficientPsLearner.download(str(self._tmp_folder),
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
        Subscribe to all relevant topics.
        """
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)

    def _init_publisher(self):
        """
        Set up the publishers as requested by the user.
        """
        if self.output_heatmap_topic is not None:
            self._instance_heatmap_publisher = rospy.Publisher(
                f'{self.output_heatmap_topic}/instance', ROS_Image, queue_size=10)
            self._semantic_heatmap_publisher = rospy.Publisher(
                f'{self.output_heatmap_topic}/semantic', ROS_Image, queue_size=10)
        if self.output_rgb_visualization_topic is not None:
            self._visualization_publisher = rospy.Publisher(self.output_rgb_visualization_topic,
                                                            ROS_Image, queue_size=10)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        rospy.init_node('opendr_efficient_panoptic_segmentation_node', anonymous=True)
        rospy.loginfo("Panoptic segmentation EfficientPS node started.")
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
                self._visualization_publisher.publish(self._bridge.to_ros_image(panoptic_image, encoding="rgb8"))

            if self._instance_heatmap_publisher is not None and self._instance_heatmap_publisher.get_num_connections() > 0:
                self._instance_heatmap_publisher.publish(self._bridge.to_ros_image(prediction[0]))
            if self._semantic_heatmap_publisher is not None and self._semantic_heatmap_publisher.get_num_connections() > 0:
                self._semantic_heatmap_publisher.publish(self._bridge.to_ros_image(prediction[1]))

        except Exception as e:
            rospy.logwarn(f'Failed to generate prediction: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_rgb_image_topic', type=str, default='/usb_cam/image_raw',
                        help='listen to RGB images on this topic')
    parser.add_argument('-oh', '--output_heatmap_topic',
                        type=lambda value: value if value.lower() != "none" else None,
                        default='/opendr/panoptic',
                        help='publish the semantic and instance maps on this topic as "OUTPUT_HEATMAP_TOPIC/semantic" \
                             and "OUTPUT_HEATMAP_TOPIC/instance"')
    parser.add_argument('-ov', '--output_rgb_image_topic',
                        type=lambda value: value if value.lower() != "none" else None,
                        default='/opendr/panoptic/rgb_visualization',
                        help='publish the panoptic segmentation map as an RGB image on this topic or a more detailed \
                              overview if using the --detailed_visualization flag')
    parser.add_argument('--detailed_visualization', action='store_true',
                        help='generate a combined overview of the input RGB image and the semantic, instance, and \
                              panoptic segmentation maps and publish it on OUTPUT_RGB_IMAGE_TOPIC')
    parser.add_argument('--checkpoint', type=str, default='cityscapes',
                        help='download pretrained models [cityscapes, kitti] or load from the provided path')
    args = parser.parse_args()

    efficient_ps_node = EfficientPsNode(args.input_rgb_image_topic,
                                        args.checkpoint,
                                        args.output_heatmap_topic,
                                        args.output_rgb_image_topic,
                                        args.detailed_visualization)
    efficient_ps_node.listen()
