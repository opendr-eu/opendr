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

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2DArray
from opendr_bridge import ROS2Bridge

from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.engine.data import Image


class FaceDetectionNode(Node):

    def __init__(self, input_rgb_image_topic="image_raw", output_rgb_image_topic="/opendr/image_faces_annotated",
                 detections_topic="/opendr/faces", device="cuda", backbone="resnet"):
        """
        Creates a ROS2 Node for face detection with Retinaface.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, no face detection message
        is published)
        :type detections_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param backbone: retinaface backbone, options are either 'mnet' or 'resnet',
        where 'mnet' detects masked faces as well
        :type backbone: str
        """
        super().__init__('opendr_face_detection_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_rgb_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.face_publisher = self.create_publisher(Detection2DArray, detections_topic, 1)
        else:
            self.face_publisher = None

        self.bridge = ROS2Bridge()

        self.face_detector = RetinaFaceLearner(backbone=backbone, device=device)
        self.face_detector.download(path=".", verbose=True)
        self.face_detector.load("retinaface_{}".format(backbone))
        self.class_names = ["face", "masked_face"]

        self.get_logger().info("Face detection retinaface node initialized.")

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run face detection
        boxes = self.face_detector.infer(image)

        if self.face_publisher is not None:
            # Publish detections in ROS message
            ros_boxes = self.bridge.to_ros_boxes(boxes)  # Convert to ROS boxes
            self.face_publisher.publish(ros_boxes)

        if self.image_publisher is not None:
            # Get an OpenCV image back
            image = image.opencv()
            # Annotate image with face detection boxes
            image = draw_bounding_boxes(image, boxes, class_names=self.class_names)
            # Convert the annotated OpenDR image to ROS2 image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_faces_annotated")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/faces")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backbone",
                        help="Retinaface backbone, options are either 'mnet' or 'resnet', where 'mnet' detects "
                             "masked faces as well",
                        type=str, default="resnet", choices=["resnet", "mnet"])
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

    face_detection_node = FaceDetectionNode(device=device, backbone=args.backbone,
                                            input_rgb_image_topic=args.input_rgb_image_topic,
                                            output_rgb_image_topic=args.output_rgb_image_topic,
                                            detections_topic=args.detections_topic)

    rclpy.spin(face_detection_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    face_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
