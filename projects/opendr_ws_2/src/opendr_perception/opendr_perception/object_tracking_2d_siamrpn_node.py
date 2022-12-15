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
import mxnet as mx

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2D
from opendr_bridge import ROS2Bridge
from opendr_interface.srv import OpenDRSingleObjectTracking

from opendr.engine.data import Image
from opendr.perception.object_tracking_2d import SiamRPNLearner


class ObjectTrackingSiamRPNNode(Node):

    def __init__(self, input_rgb_image_topic="/image_raw",
                 output_rgb_image_topic="/opendr/image_tracking_annotated",
                 tracker_topic="/opendr/tracked_object",
                 device="cuda"):
        """
        Creates a ROS2 Node for object tracking with SiamRPN.
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
        super().__init__('object_tracking_2d_siamrpn_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_rgb_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.image_publisher = None

        if tracker_topic is not None:
            self.object_publisher = self.create_publisher(Detection2D, tracker_topic, 1)
        else:
            self.object_publisher = None

        self.bridge = ROS2Bridge()

        # Initialize the object detector
        self.tracker = SiamRPNLearner(device=device)
        self.image = None
        self.initialized = False

        self.create_service(OpenDRSingleObjectTracking, "/opendr/siamrpn_tracking_srv", self.init_box_srv_callback)

        self.get_logger().info("Object Tracking 2D SiamRPN node initialized.")

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        self.image = image

        # Run object detection
        if self.initialized:
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
                # Convert the annotated OpenDR image to ROS2 image message using bridge and publish it
                self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))

    def init_box_srv_callback(self, request):
        # combine incoming box with current image to initialize tracker
        self.initialized = False
        init_box = self.bridge.from_ros_single_tracking_annotation(request.init_box)
        self.tracker.infer(self.image, init_box)
        self.initialized = True
        return True


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/image_raw")
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

    object_tracker_2d_siamrpn_node = ObjectTrackingSiamRPNNode(device=device,
                                                               input_rgb_image_topic=args.input_rgb_image_topic,
                                                               output_rgb_image_topic=args.output_rgb_image_topic,
                                                               tracker_topic=args.tracker_topic)

    rclpy.spin(object_tracker_2d_siamrpn_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_tracker_2d_siamrpn_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
