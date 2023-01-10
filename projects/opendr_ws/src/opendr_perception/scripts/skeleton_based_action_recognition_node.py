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

import argparse
import rospy
import torch
import numpy as np
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis
from opendr_bridge.msg import OpenDRPose2D
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.pose_estimation import draw
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner
from opendr.perception.skeleton_based_action_recognition import ProgressiveSpatioTemporalGCNLearner
from opendr.engine.data import Image


class SkeletonActionRecognitionNode:

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw",
                 output_rgb_image_topic="/opendr/image_pose_annotated",
                 pose_annotations_topic="/opendr/poses",
                 output_category_topic="/opendr/skeleton_recognized_action",
                 output_category_description_topic="/opendr/skeleton_recognized_action_description",
                 device="cuda", model='stgcn'):
        """
        Creates a ROS Node for skeleton-based action recognition
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_rgb_image_topic: str
        :param pose_annotations_topic: Topic to which we are publishing the annotations (if None, we are not publishing
        annotated pose annotations)
        :type pose_annotations_topic:  str
        :param output_category_topic: Topic to which we are publishing the recognized action category info
        (if None, we are not publishing the info)
        :type output_category_topic: str
        :param output_category_description_topic: Topic to which we are publishing the description of the recognized
        action (if None, we are not publishing the description)
        :type output_category_description_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model:  model to use for skeleton-based action recognition.
         (Options: 'stgcn', 'pstgcn')
        :type model: str
        """

        # Set up ROS topics and bridge
        self.input_rgb_image_topic = input_rgb_image_topic
        self.bridge = ROSBridge()

        if output_rgb_image_topic is not None:
            self.image_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=1)
        else:
            self.image_publisher = None

        if pose_annotations_topic is not None:
            self.pose_publisher = rospy.Publisher(pose_annotations_topic, OpenDRPose2D, queue_size=1)
        else:
            self.pose_publisher = None

        if output_category_topic is not None:
            self.hypothesis_publisher = rospy.Publisher(output_category_topic, ObjectHypothesis, queue_size=1)
        else:
            self.hypothesis_publisher = None

        if output_category_description_topic is not None:
            self.string_publisher = rospy.Publisher(output_category_description_topic, String, queue_size=1)
        else:
            self.string_publisher = None

        # Initialize the pose estimation
        self.pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=2,
                                                         mobilenet_use_stride=False,
                                                         half_precision=False
                                                         )
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")

        # Initialize the skeleton_based action recognition
        if model == 'stgcn':
            self.action_classifier = SpatioTemporalGCNLearner(device=device, dataset_name='nturgbd_cv',
                                                              method_name=model, in_channels=2, num_point=18,
                                                              graph_type='openpose')
        elif model == 'pstgcn':
            self.action_classifier = ProgressiveSpatioTemporalGCNLearner(device=device, dataset_name='nturgbd_cv',
                                                                         topology=[5, 4, 5, 2, 3, 4, 3, 4],
                                                                         in_channels=2, num_point=18,
                                                                         graph_type='openpose')

        model_saved_path = self.action_classifier.download(path="./pretrained_models/"+model,
                                                           method_name=model, mode="pretrained",
                                                           file_name=model+'_ntu_cv_lw_openpose')
        self.action_classifier.load(model_saved_path, model+'_ntu_cv_lw_openpose')

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_skeleton_action_recognition_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Skeleton-based action recognition node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run pose estimation
        poses = self.pose_estimator.infer(image)
        if len(poses) > 2:
            # select two poses with highest energy
            poses = _select_2_poses(poses)

        # Get an OpenCV image back
        image = image.opencv()
        #  Annotate image and publish results
        for pose in poses:
            if self.pose_publisher is not None:
                ros_pose = self.bridge.to_ros_pose(pose)
                self.pose_publisher.publish(ros_pose)
                # We get can the data back using self.bridge.from_ros_pose(ros_pose)
                # e.g., opendr_pose = self.bridge.from_ros_pose(ros_pose)
                draw(image, pose)

        if self.image_publisher is not None:
            message = self.bridge.to_ros_image(Image(image), encoding='bgr8')
            self.image_publisher.publish(message)

        num_frames = 300
        poses_list = []
        for _ in range(num_frames):
            poses_list.append(poses)
        skeleton_seq = _pose2numpy(num_frames, poses_list)

        # Run action recognition
        category = self.action_classifier.infer(skeleton_seq)
        category.confidence = float(category.confidence.max())

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
    T = 300
    V = 18
    M = 2  # num_person_in
    skeleton_seq = np.zeros((1, C, T, V, M))
    for t in range(num_current_frames):
        for m in range(len(poses_list[t])):
            skeleton_seq[0, 0:2, t, :, m] = np.transpose(poses_list[t][m].data)
    return skeleton_seq


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_pose_annotated")
    parser.add_argument("-p", "--pose_annotations_topic", help="Topic name for pose annotations",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/poses")
    parser.add_argument("-c", "--output_category_topic", help="Topic name for recognized action category",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/skeleton_recognized_action")
    parser.add_argument("-d", "--output_category_description_topic",
                        help="Topic name for description of the recognized action category",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/skeleton_recognized_action_description")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", help="Model to use, either \"stgcn\" or \"pstgcn\"",
                        type=str, default="stgcn", choices=["stgcn", "pstgcn"])

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

    skeleton_action_recognition_node = \
        SkeletonActionRecognitionNode(input_rgb_image_topic=args.input_rgb_image_topic,
                                      output_rgb_image_topic=args.output_rgb_image_topic,
                                      pose_annotations_topic=args.pose_annotations_topic,
                                      output_category_topic=args.output_category_topic,
                                      output_category_description_topic=args.output_category_description_topic,
                                      device=device,
                                      model=args.model)
    skeleton_action_recognition_node.listen()
