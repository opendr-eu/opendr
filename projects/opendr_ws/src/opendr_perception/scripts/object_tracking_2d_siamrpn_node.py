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
import mxnet as mx

import cv2
from math import dist
import rospy

from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2D
from opendr_bridge import ROSBridge

from opendr.engine.data import Image
from opendr.engine.target import TrackingAnnotation, BoundingBox
from opendr.perception.object_tracking_2d import SiamRPNLearner
from opendr.perception.object_detection_2d import YOLOv3DetectorLearner


class ObjectTrackingSiamRPNNode:
    def __init__(self, object_detector, input_rgb_image_topic="/usb_cam/image_raw",
                 output_rgb_image_topic="/opendr/image_tracking_annotated",
                 tracker_topic="/opendr/tracked_object",
                 device="cuda"):
        """
        Creates a ROS Node for object tracking with SiamRPN.
        :param object_detector: An object detector learner to use for initialization
        :type object_detector: opendr.engine.learners.Learner
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param tracker_topic: Topic to which we are publishing the annotation
        :type tracker_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        self.input_rgb_image_topic = input_rgb_image_topic

        if output_rgb_image_topic is not None:
            self.image_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=1)
        else:
            self.image_publisher = None

        if tracker_topic is not None:
            self.object_publisher = rospy.Publisher(tracker_topic, Detection2D, queue_size=1)
        else:
            self.object_publisher = None

        self.bridge = ROSBridge()

        self.object_detector = object_detector
        # Initialize the object detector
        self.tracker = SiamRPNLearner(device=device)
        self.image = None
        self.initialized = False

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_object_tracking_2d_siamrpn_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.img_callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Object Tracking 2D SiamRPN node started.")
        rospy.spin()

    def img_callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        self.image = image

        if not self.initialized:
            # Run object detector to initialize the tracker
            image = self.bridge.from_ros_image(data, encoding='bgr8')
            boxes = self.object_detector.infer(image)

            img_center = [int(image.data.shape[2] // 2), int(image.data.shape[1] // 2)]  # width, height
            # Find the box that is closest to the center of the image
            center_box = BoundingBox("", left=0, top=0, width=0, height=0)
            min_distance = dist([center_box.left, center_box.top], img_center)
            for box in boxes:
                new_distance = dist([int(box.left + box.width // 2), int(box.top + box.height // 2)], img_center)
                if new_distance < min_distance:
                    center_box = box
                    min_distance = dist([center_box.left, center_box.top], img_center)

            # Initialize tracker with the most central box found
            init_box = TrackingAnnotation(center_box.name,
                                          center_box.left, center_box.top, center_box.width, center_box.height,
                                          id=0, score=center_box.confidence)

            self.tracker.infer(self.image, init_box)
            self.initialized = True
            rospy.loginfo("Object Tracking 2D SiamRPN node initialized with the most central bounding box.")

        if self.initialized:
            # Run object tracking
            box = self.tracker.infer(image)

            if self.object_publisher is not None:
                # Publish detections in ROS message
                ros_boxes = self.bridge.to_ros_single_tracking_annotation(box)
                self.object_publisher.publish(ros_boxes)

            if self.image_publisher is not None:
                # Get an OpenCV image back
                image = image.opencv()
                cv2.rectangle(image, (box.left, box.top),
                              (box.left + box.width, box.top + box.height),
                              (0, 255, 255), 3)
                # Convert the annotated OpenDR image to ROS image message using bridge and publish it
                self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_tracking_annotated")
    parser.add_argument("-t", "--tracker_topic", help="Topic name for tracker messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/tracked_object")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    try:
        if args.device == "cuda" and mx.context.num_gpus() > 0:
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

    object_detector = YOLOv3DetectorLearner(backbone="darknet53", device=device)
    object_detector.download(path=".", verbose=True)
    object_detector.load("yolo_default")

    object_tracker_2d_siamrpn_node = ObjectTrackingSiamRPNNode(object_detector=object_detector, device=device,
                                                               input_rgb_image_topic=args.input_rgb_image_topic,
                                                               output_rgb_image_topic=args.output_rgb_image_topic,
                                                               tracker_topic=args.tracker_topic)
    object_tracker_2d_siamrpn_node.listen()


if __name__ == '__main__':
    main()
