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
import numpy as np
from time import perf_counter

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from vision_msgs.msg import ObjectHypothesis
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROS2Bridge
from opendr_interface.msg import OpenDRPose2D

from opendr.engine.data import Image
from opendr.perception.pose_estimation import draw
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.skeleton_based_action_recognition import CoSTGCNLearner


class CoSkeletonActionRecognitionNode(Node):

    def __init__(self, input_rgb_image_topic="image_raw",
                 output_rgb_image_topic="/opendr/image_pose_annotated",
                 pose_annotations_topic="/opendr/poses",
                 output_category_topic="/opendr/continual_skeleton_recognized_action",
                 output_category_description_topic="/opendr/continual_skeleton_recognized_action_description",
                 performance_topic=None,
                 device="cuda", model='costgcn'):
        """
        Creates a ROS2 Node for continual skeleton-based action recognition.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, we are not
        publishing annotated image)
        :type output_rgb_image_topic: str
        :param pose_annotations_topic: Topic to which we are publishing the pose annotations (if None, we are not publishing
        annotated pose annotations)
        :type pose_annotations_topic:  str
        :param output_category_topic: Topic to which we are publishing the recognized action category info
        (if None, we are not publishing the info)
        :type output_category_topic: str
        :param output_category_description_topic: Topic to which we are publishing the description of the recognized
        action (if None, we are not publishing the description)
        :type output_category_description_topic:  str
        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model:  model to use for skeleton-based action recognition.
         (Options: "costgcn")
        :type model: str
        """
        super().__init__('opendr_continual_skeleton_based_action_recognition_node')
        # Set up ROS topics and bridge

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_rgb_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.image_publisher = None

        if pose_annotations_topic is not None:
            self.pose_publisher = self.create_publisher(OpenDRPose2D, pose_annotations_topic, 1)
        else:
            self.pose_publisher = None

        if output_category_topic is not None:
            self.hypothesis_publisher = self.create_publisher(ObjectHypothesis, output_category_topic, 1)
        else:
            self.hypothesis_publisher = None

        if output_category_description_topic is not None:
            self.string_publisher = self.create_publisher(String, output_category_description_topic, 1)
        else:
            self.string_publisher = None

        if performance_topic is not None:
            self.performance_publisher = self.create_publisher(Float32, performance_topic, 1)
        else:
            self.performance_publisher = None

        self.bridge = ROS2Bridge()

        # Initialize the pose estimation
        self.pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=2,
                                                         mobilenet_use_stride=False,
                                                         half_precision=False
                                                         )
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")

        # Initialize the skeleton_based action recognition
        self.action_classifier = CoSTGCNLearner(device=device, backbone=model, in_channels=2, num_point=18,
                                                graph_type='openpose')

        model_saved_path = self.action_classifier.download(method_name='stgcn/stgcn_ntu_cv_lw_openpose',
                                                           mode="pretrained",
                                                           file_name="stgcn_ntu_cv_lw_openpose.pt")
        self.action_classifier.load(model_saved_path)
        self.get_logger().info("Skeleton-based action recognition node initialized.")

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        if self.performance_publisher:
            start_time = perf_counter()

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run pose estimation
        poses = self.pose_estimator.infer(image)
        if len(poses) > 2:
            # select two poses with the highest energy
            poses = _select_2_poses(poses)

        num_frames = 1
        poses_list = [poses]
        skeleton_seq = _pose2numpy(num_frames, poses_list)

        # Run action recognition
        result = self.action_classifier.infer(skeleton_seq)  # input_size: BxCxTxVxS
        category = result[0]
        category.confidence = float(category.confidence.max())

        if self.performance_publisher:
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)  # NOQA
            fps_msg = Float32()
            fps_msg.data = fps
            self.performance_publisher.publish(fps_msg)

        #  Publish detections in ROS message
        if self.pose_publisher is not None:
            for pose in poses:
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

        if self.hypothesis_publisher is not None:
            self.hypothesis_publisher.publish(self.bridge.to_ros_category(category))

        if self.string_publisher is not None:
            self.string_publisher.publish(self.bridge.to_ros_category_description(category))


def _select_2_poses(poses):
    selected_poses = []
    energy = []
    for i in range(len(poses)):
        s = poses[i].data[:, 0].std() + poses[i].data[:, 1].std()
        energy.append(s)
    energy = np.array(energy)
    index = energy.argsort()[::-1][0:2]
    for i in range(len(index)):
        selected_poses.append(poses[index[i]])
    return selected_poses


def _pose2numpy(num_current_frames, poses_list):
    C = 2
    V = 18
    M = 2  # num_person_in
    skeleton_seq = np.zeros((1, C, num_current_frames, V, M))
    for t in range(num_current_frames):
        for m in range(len(poses_list[t])):
            skeleton_seq[0, 0:2, t, :, m] = np.transpose(poses_list[t][m].data)
    return torch.tensor(skeleton_seq)


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input image",
                        type=str, default="image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_pose_annotated")
    parser.add_argument("-p", "--pose_annotations_topic", help="Topic name for pose annotations",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/poses")
    parser.add_argument("-c", "--output_category_topic", help="Topic name for recognized action category",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/continual_skeleton_recognized_action")
    parser.add_argument("-d", "--output_category_description_topic", help="Topic name for description of the "
                                                                          "recognized action category",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/continual_skeleton_recognized_action_description")
    parser.add_argument("--performance_topic", help="Topic name for performance messages, disabled (None) by default",
                        type=str, default=None)
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", help="Model to use, either \"costgcn\"",
                        type=str, default="costgcn", choices=["costgcn"])

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

    continual_skeleton_action_recognition_node = \
        CoSkeletonActionRecognitionNode(input_rgb_image_topic=args.input_rgb_image_topic,
                                        output_rgb_image_topic=args.output_rgb_image_topic,
                                        pose_annotations_topic=args.pose_annotations_topic,
                                        output_category_topic=args.output_category_topic,
                                        output_category_description_topic=args.output_category_description_topic,
                                        performance_topic=args.performance_topic,
                                        device=device,
                                        model=args.model)

    rclpy.spin(continual_skeleton_action_recognition_node)
    continual_skeleton_action_recognition_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
