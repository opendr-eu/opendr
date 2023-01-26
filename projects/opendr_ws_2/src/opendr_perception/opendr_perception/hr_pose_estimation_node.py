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
import torch

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROS2Bridge
from opendr_interface.msg import OpenDRPose2D

from opendr.engine.data import Image
from opendr.perception.pose_estimation import draw
from opendr.perception.pose_estimation import HighResolutionPoseEstimationLearner


class PoseEstimationNode(Node):

    def __init__(self, input_rgb_image_topic="image_raw", output_rgb_image_topic="/opendr/image_pose_annotated",
                 detections_topic="/opendr/poses", device="cuda",
                 num_refinement_stages=2, use_stride=False, half_precision=False):
        """
        Creates a ROS2 Node for pose estimation with Lightweight OpenPose.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, no pose detection message
        is published)
        :type detections_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param num_refinement_stages: Specifies the number of pose estimation refinement stages are added on the
        model's head, including the initial stage. Can be 0, 1 or 2, with more stages meaning slower and more accurate
        inference
        :type num_refinement_stages: int
        :param use_stride: Whether to add a stride value in the model, which reduces accuracy but increases
        inference speed
        :type use_stride: bool
        :param half_precision: Enables inference using half (fp16) precision instead of single (fp32) precision.
        Valid only for GPU-based inference
        :type half_precision: bool
        """
        super().__init__('opendr_pose_estimation_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_rgb_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.pose_publisher = self.create_publisher(OpenDRPose2D, detections_topic, 1)
        else:
            self.pose_publisher = None

        self.bridge = ROS2Bridge()

        # Initialize the high resolution pose estimation learner
        self.pose_estimator = HighResolutionPoseEstimationLearner(device=device, num_refinement_stages=num_refinement_stages,
                                                                  mobilenet_use_stride=use_stride,
                                                                  half_precision=half_precision)
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")

        self.get_logger().info("Pose estimation node initialized.")

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run pose estimation
        poses = self.pose_estimator.infer(image)

        #  Publish detections in ROS message
        if self.pose_publisher is not None:
            for pose in poses:
                if pose.id is None:  # Temporary fix for pose not having id
                    pose.id = -1
                if self.pose_publisher is not None:
                    # Convert OpenDR pose to ROS2 pose message using bridge and publish it
                    self.pose_publisher.publish(self.bridge.to_ros_pose(pose))

        if self.image_publisher is not None:
            # Get an OpenCV image back
            image = image.opencv()
            # Annotate image with poses
            for pose in poses:
                draw(image, pose)
            # Convert the annotated OpenDR image to ROS2 image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image, if \"None\" "
                                                               "no output image is published",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_pose_annotated")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages, if \"None\" "
                                                         "no detection message is published",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/poses")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    args = parser.parse_args()

    try:
        if args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif args.device == "cuda":
            print("GPU not found. Using CPU instead.")
            device = "cpu"
        else:
            print("Using CPU.")
            device = "cpu"
    except:
        print("Using CPU.")
        device = "cpu"

    if args.accelerate:
        stride = True
        stages = 0
        half_prec = True
    else:
        stride = False
        stages = 2
        half_prec = False

    pose_estimator_node = PoseEstimationNode(device=device,
                                             input_rgb_image_topic=args.input_rgb_image_topic,
                                             output_rgb_image_topic=args.output_rgb_image_topic,
                                             detections_topic=args.detections_topic,
                                             num_refinement_stages=stages, use_stride=stride, half_precision=half_prec)

    rclpy.spin(pose_estimator_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_estimator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
