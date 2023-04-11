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
import torch
from numpy import std, ndarray
import cv2
from time import perf_counter

import rospy
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2D
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge.msg import OpenDRPose2D
from opendr_bridge import ROSBridge

from opendr.engine.data import Image
from opendr.engine.target import BoundingBox
from opendr.perception.pose_estimation import get_bbox
from opendr.perception.pose_estimation import LightweightOpenPoseLearner


class WaveDetectionNode:

    def __init__(self, input_rgb_image_topic=None,
                 input_pose_topic="/opendr/poses",
                 output_rgb_image_topic="/opendr/image_pose_annotated",
                 detections_topic="/opendr/wave",
                 performance_topic=None,
                 device="cuda",
                 num_refinement_stages=2, use_stride=False, half_precision=False):
        """
        Creates a ROS Node for rule-based wave detection via pose estimation.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param input_pose_topic: Topic from which we are reading the input pose list
        :type input_pose_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, no wave detection message
        is published)
        :type detections_topic:  str
        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic:  str
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

            if performance_topic is not None:
                self.image_performance_publisher = rospy.Publisher(performance_topic + "/image", Float32, queue_size=1)
            else:
                self.image_performance_publisher = None
        else:
            self.input_rgb_image_topic = None
            self.image_publisher = None
            self.pose_estimator = None

        if input_pose_topic is not None:
            self.input_pose_topic = input_pose_topic

            if performance_topic is not None:
                self.wave_performance_publisher = rospy.Publisher(performance_topic + "/wave", Float32, queue_size=1)
            else:
                self.wave_performance_publisher = None
        else:
            self.input_pose_topic = None

        self.wave_publisher = rospy.Publisher(detections_topic, Detection2D, queue_size=1)

        self.bridge = ROSBridge()
        self.pose_list = []

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_wave_detection_node', anonymous=True)
        if self.input_rgb_image_topic is not None:
            rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.image_callback, queue_size=1, buff_size=10000000)
        if self.input_pose_topic is not None:
            rospy.Subscriber(self.input_pose_topic, OpenDRPose2D, self.wave_callback, queue_size=1, buff_size=10000000)

        if self.input_pose_topic and not self.input_rgb_image_topic:
            rospy.loginfo("Wave detection node started in detection mode.")
        elif self.input_rgb_image_topic and not self.input_pose_topic:
            rospy.loginfo("Wave detection node started in visualization mode.")
        elif self.input_pose_topic and self.input_rgb_image_topic:
            rospy.loginfo("Wave detection node started in both detection and visualization mode.")

        rospy.spin()

    def wave_callback(self, data):
        """
        Callback that processes the input pose data and publishes to the corresponding topics.
        :param data: Input pose message
        :type data: opendr_bridge.msg.OpenDRPose2D
        """
        if self.wave_performance_publisher:
            start_time = perf_counter()
        poses = [self.bridge.from_ros_pose(data)]
        # convert to dict with pose id as key for convenience
        poses_dict = {k: v for k, v in zip([poses[i].id for i in range(len(poses))], poses)}

        if len(poses) > 0:
            self.pose_list.append(poses_dict)
            if len(self.pose_list) > 15:
                self.pose_list = self.pose_list[1:]

            pose_waves = self.wave_detection()

            for pose_id, waving_and_pose in pose_waves.items():
                waving = waving_and_pose[0]
                pose = waving_and_pose[1]
                x, y, w, h = get_bbox(pose)

                if self.wave_performance_publisher:
                    end_time = perf_counter()
                    fps = 1.0 / (end_time - start_time)  # NOQA
                    fps_msg = Float32()
                    fps_msg.data = fps
                    self.wave_performance_publisher.publish(fps_msg)

                self.wave_publisher.publish(self.bridge.to_ros_box(BoundingBox(left=x, top=y, width=w, height=h,
                                                                               name=waving, score=pose.confidence)))

    def image_callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        if self.image_performance_publisher:
            start_time = perf_counter()
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run pose estimation
        poses = self.pose_estimator.infer(image)
        # convert to dict with pose id as key for convenience
        poses_dict = {k: v for k, v in zip([poses[i].id for i in range(len(poses))], poses)}

        if len(poses) > 0:
            self.pose_list.append(poses_dict)
            if len(self.pose_list) > 15:
                self.pose_list = self.pose_list[1:]

            pose_waves = self.wave_detection()

            for pose_id, waving_and_pose in pose_waves.items():
                waving = waving_and_pose[0]
                pose = waving_and_pose[1]
                x, y, w, h = get_bbox(pose)

                if self.image_performance_publisher:
                    end_time = perf_counter()
                    fps = 1.0 / (end_time - start_time)  # NOQA
                    fps_msg = Float32()
                    fps_msg.data = fps
                    self.image_performance_publisher.publish(fps_msg)

                if waving == 1:
                    if type(image) != ndarray:
                        # Get an OpenCV image back
                        image = image.opencv()
                    # Paint person bounding box inferred from pose
                    color = (0, 0, 255)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, "Waving person", (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 2, cv2.LINE_AA)

                self.wave_publisher.publish(self.bridge.to_ros_box(BoundingBox(left=x, top=y, width=w, height=h,
                                                                               name=waving, score=pose.confidence)))

        if type(image) != ndarray:
            # Get an OpenCV image back
            image = image.opencv()
        # Convert the annotated OpenDR image to ROS image message using bridge and publish it
        self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))

    def wave_detection(self):
        """
        This method detects waving gestures based on poses.

        It works on a "pose_list" which contains dictionaries of poses for past frames.
        Each dictionary uses the pose ID as key and the actual pose as value. As such, grabbing one pose ID key and looping
        backwards in the pose_list, the method can infer information about the pose over several frames.

        Waving is detected when the left or right wrist is detected above neck height and is moving along the x-axis.
        Movement is detected via the standard deviation of the wrist's x position.
        """
        pose_waves = {}  # {pose_id: [waving, pose]} (waving = 1, not waving = 0, can't detect = -1)
        # Loop through pose ids in last frame to check for waving in each one
        for pose_id in self.pose_list[-1].keys():
            pose_waves[pose_id] = [0, self.pose_list[-1][pose_id]]
            # Get average position of wrists, get list of wrists positions on x-axis
            r_wri_avg_pos = [0, 0]
            l_wri_avg_pos = [0, 0]
            r_wri_x_positions = []
            l_wri_x_positions = []
            for frame in self.pose_list:
                try:
                    if frame[pose_id]["r_wri"][0] != -1:
                        r_wri_avg_pos += frame[pose_id]["r_wri"]
                        r_wri_x_positions.append(frame[pose_id]["r_wri"][0])
                    if frame[pose_id]["l_wri"][0] != -1:
                        l_wri_avg_pos += frame[pose_id]["l_wri"]
                        l_wri_x_positions.append(frame[pose_id]["l_wri"][0])
                except KeyError:  # Couldn't find this pose_id in previous frames
                    pose_waves[pose_id][0] = -1
                    continue
            r_wri_avg_pos = [r_wri_avg_pos[0] / len(self.pose_list), r_wri_avg_pos[1] / len(self.pose_list)]
            l_wri_avg_pos = [l_wri_avg_pos[0] / len(self.pose_list), l_wri_avg_pos[1] / len(self.pose_list)]
            r_wri_x_positions = [r_wri_x_positions[i] - r_wri_avg_pos[0] for i in range(len(r_wri_x_positions))]
            l_wri_x_positions = [l_wri_x_positions[i] - l_wri_avg_pos[0] for i in range(len(l_wri_x_positions))]

            pose_ = None  # NOQA
            if len(self.pose_list) > 0:
                pose_ = self.pose_list[-1][pose_id]
            else:
                pose_waves[pose_id][0] = -1
                continue
            nose_height, neck_height = pose_["nose"][1], pose_["neck"][1]
            r_wri_height, l_wri_height = r_wri_avg_pos[1], l_wri_avg_pos[1]

            if nose_height == -1 or neck_height == -1:
                # Can't detect upper pose_ (neck-nose), can't assume waving
                pose_waves[pose_id][0] = -1
                continue
            if r_wri_height == 0 and l_wri_height == 0:
                # Can't detect wrists, can't assume waving
                pose_waves[pose_id][0] = -1
                continue

            # Calculate the standard deviation threshold based on the distance between neck and nose to get proportions
            # The farther away the pose is the smaller the threshold, as the standard deviation would be smaller due to
            # the smaller pose
            distance = neck_height - nose_height
            std_threshold = 5 + ((distance - 50) / (200 - 50)) * 10

            # Check for wrist movement over multiple frames
            # Wrist movement is determined from wrist x position standard deviation
            r_wrist_movement_detected = False
            l_wrist_movement_detected = False
            r_wri_x_pos_std, l_wri_x_pos_std = 0, 0  # NOQA
            if r_wri_height < neck_height:
                if len(r_wri_x_positions) > len(self.pose_list) / 2:
                    r_wri_x_pos_std = std(r_wri_x_positions)
                    if r_wri_x_pos_std > std_threshold:
                        r_wrist_movement_detected = True
            if l_wri_height < neck_height:
                if len(l_wri_x_positions) > len(self.pose_list) / 2:
                    l_wri_x_pos_std = std(l_wri_x_positions)
                    if l_wri_x_pos_std > std_threshold:
                        l_wrist_movement_detected = True

            if r_wrist_movement_detected:
                pose_waves[pose_id][0] = 1
            elif l_wrist_movement_detected:
                pose_waves[pose_id][0] = 1

        return pose_waves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--input_pose_topic", help="Topic name for input pose list",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/poses")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=str, default="/opendr/wave")
    parser.add_argument("-ii", "--input_rgb_image_topic", help="Topic name for input rgb image, used for visualization",
                        type=str, default=None)
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=str, default="/opendr/image_wave_annotated")
    parser.add_argument("--performance_topic", help="Topic name for performance messages, disabled (None) by default",
                        type=str, default=None)
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

    wave_estimator_node = WaveDetectionNode(device=device,
                                            input_rgb_image_topic=args.input_rgb_image_topic,
                                            input_pose_topic=args.input_pose_topic,
                                            output_rgb_image_topic=args.output_rgb_image_topic,
                                            detections_topic=args.detections_topic,
                                            performance_topic=args.performance_topic,
                                            num_refinement_stages=stages, use_stride=stride, half_precision=half_prec)
    wave_estimator_node.listen()


if __name__ == '__main__':
    main()
