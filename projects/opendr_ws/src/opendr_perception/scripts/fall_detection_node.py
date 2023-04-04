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

import cv2
import argparse
import torch

import rospy
from vision_msgs.msg import Detection2D
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr_bridge.msg import OpenDRPose2D

from opendr.engine.data import Image
from opendr.engine.target import BoundingBox
from opendr.perception.pose_estimation import get_bbox
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.fall_detection import FallDetectorLearner


class FallDetectionNode:

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw", input_pose_topic="/opendr/poses",
                 output_rgb_image_topic="/opendr/image_fallen_annotated", detections_topic="/opendr/fallen",
                 device="cuda", num_refinement_stages=2, use_stride=False, half_precision=False):
        """
        Creates a ROS Node for rule-based fall detection via pose estimation.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param input_pose_topic: Topic from which we are reading the input pose list
        :type input_pose_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, no fall detection message
        is published)
        :type detections_topic:  str
        :param device: device on which we are running pose estimation inference ('cpu' or 'cuda')
        :type device: str
        :param num_refinement_stages: Specifies the number of pose estimation refinement stages are added on the
        model's head, including the initial stage. Can be 0, 1 or 2, with more stages meaning slower and more accurate
        inference
        :type num_refinement_stages: int
        :param use_stride: Whether to add a stride value in the pose estimation model,
        which reduces accuracy but increases inference speed
        :type use_stride: bool
        :param half_precision: Enables pose estimation inference using half (fp16) precision instead of
        single (fp32) precision. Valid only for GPU-based inference
        :type half_precision: bool
        """
        # If input image topic is set, it is used for visualization
        if input_rgb_image_topic is not None:
            self.input_rgb_image_topic = input_rgb_image_topic
            self.image_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=1)

            # Initialize the pose estimation learner needed to run pose estimation on the image
            self.pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=num_refinement_stages,
                                                             mobilenet_use_stride=use_stride,
                                                             half_precision=half_precision)
            self.pose_estimator.download(path=".", verbose=True)
            self.pose_estimator.load("openpose_default")
        else:
            self.input_rgb_image_topic = None
            self.image_publisher = None
            self.pose_estimator = None

        if input_pose_topic is not None:
            self.input_pose_topic = input_pose_topic
        else:
            self.input_pose_topic = None

        self.fall_publisher = rospy.Publisher(detections_topic, Detection2D, queue_size=1)

        self.bridge = ROSBridge()

        # Initialize the fall detection learner
        self.fall_detector = FallDetectorLearner(self.pose_estimator)

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_fall_detection_node', anonymous=True)
        if self.input_rgb_image_topic is not None:
            rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.image_callback, queue_size=1, buff_size=10000000)
        if self.input_pose_topic is not None:
            rospy.Subscriber(self.input_pose_topic, OpenDRPose2D, self.pose_callback, queue_size=1, buff_size=10000000)

        if self.input_pose_topic and not self.input_rgb_image_topic:
            rospy.loginfo("Fall detection node started in detection mode.")
        elif self.input_rgb_image_topic and not self.input_pose_topic:
            rospy.loginfo("Fall detection node started in visualization mode.")
        elif self.input_pose_topic and self.input_rgb_image_topic:
            rospy.loginfo("Fall detection node started in both detection and visualization mode.")

        rospy.spin()

    def pose_callback(self, data):
        """
        Callback that processes the input pose data and publishes to the corresponding topics.
        :param data: Input pose message
        :type data: opendr_bridge.msg.OpenDRPose2D
        """
        poses = [self.bridge.from_ros_pose(data)]
        x, y, w, h = get_bbox(poses[0])  # Get bounding box for pose
        detections = self.fall_detector.infer(poses)  # Run fall detection
        fallen = detections[0][0].data  # Class: 1 = fallen, -1 = standing, 0 = can't detect

        # Create Detection2D that contains the bbox of the pose as well as the detection class
        ros_detection = self.bridge.to_ros_box(BoundingBox(left=x, top=y, width=w, height=h,
                                                           name=fallen, score=poses[0].confidence))
        self.fall_publisher.publish(ros_detection)

    def image_callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run fall detection
        detections = self.fall_detector.infer(image)

        # Get an OpenCV image back
        image = image.opencv()

        for detection in detections:
            fallen = detection[0].data
            pose = detection[1]
            x, y, w, h = get_bbox(pose)

            if fallen == 1:
                if self.image_publisher is not None:
                    # Paint person bounding box inferred from pose
                    color = (0, 0, 255)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, "Fallen person", (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 2, cv2.LINE_AA)

            # Create Detection2D that contains the bbox of the pose as well as the detection class
            ros_detection = self.bridge.to_ros_box(BoundingBox(left=x, top=y, width=w, height=h,
                                                               name=fallen, score=pose.confidence))
            self.fall_publisher.publish(ros_detection)
        self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--input_pose_topic", help="Topic name for input pose list",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/poses")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=str, default="/opendr/fallen")
    parser.add_argument("-ii", "--input_rgb_image_topic", help="Topic name for input rgb image, used for visualization",
                        type=str, default=None)
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=str, default="/opendr/image_fallen_annotated")
    parser.add_argument("--device", help="Device to use for pose estimation which runs when an input_rgb_image_topic "
                                         "is provided, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--accelerate", help="Enables acceleration flags for pose estimation which runs when an "
                                             "input_rgb_image_topic is provided (e.g., stride)",
                        default=False, action="store_true")
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

    if args.input_rgb_image_topic is None and args.input_pose_topic is None:
        raise ValueError("Please provide an input_pose_topic and/or input_rgb_image_topic.")

    fall_detection_node = FallDetectionNode(device=device,
                                            input_rgb_image_topic=args.input_rgb_image_topic,
                                            input_pose_topic=args.input_pose_topic,
                                            output_rgb_image_topic=args.output_rgb_image_topic,
                                            detections_topic=args.detections_topic,
                                            num_refinement_stages=stages, use_stride=stride, half_precision=half_prec)
    fall_detection_node.listen()


if __name__ == '__main__':
    main()
