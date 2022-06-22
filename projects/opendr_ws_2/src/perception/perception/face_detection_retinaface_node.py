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

import rclpy
from rclpy.node import Node

import cv2
import mxnet as mx
import numpy as np

from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2DArray
from ros2_bridge.bridge import ROS2Bridge

from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.engine.data import Image


class FaceDetectionNode(Node):

    def __init__(self, input_image_topic="image_raw", output_image_topic="/opendr/image_boxes_annotated",
                 face_detections_topic="/opendr/faces", device="cuda", backbone="resnet"):
        super().__init__('face_detection_node')

        if output_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_image_topic, 1)
        else:
            self.image_publisher = None

        if face_detections_topic is not None:
            self.face_publisher = self.create_publisher(Detection2DArray, face_detections_topic, 1)
        else:
            self.face_publisher = None
        # Setting queue size to 1 drops frames during forward pass, thus removing delay
        queue_size = 1
        self.image_subscriber = self.create_subscription(ROS_Image, input_image_topic, self.callback, queue_size)

        self.bridge = ROS2Bridge()

        self.face_detector = RetinaFaceLearner(backbone=backbone, device=device)
        self.face_detector.download(path=".", verbose=True)
        self.face_detector.load("retinaface_{}".format(backbone))
        self.class_names = ["face", "masked_face"]

    def callback(self, data):
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        cv2.imshow("image", image.opencv())
        cv2.waitKey(1)

        boxes = self.face_detector.infer(image)

        image = image.opencv()

        ros_boxes = self.bridge.to_ros_boxes(boxes)
        if self.face_publisher is not None:
            self.face_publisher.publish(ros_boxes)

        # Annotate image and publish result
        odr_boxes = self.bridge.from_ros_boxes(ros_boxes)
        image = draw_bounding_boxes(image, odr_boxes, class_names=self.class_names)
        if self.image_publisher is not None:
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    try:
        if mx.context.num_gpus() > 0:
            print("GPU found.")
            device = 'cuda'
        else:
            print("GPU not found. Using CPU instead.")
            device = 'cpu'
    except:
        device = 'cpu'

    face_detection_node = FaceDetectionNode(device=device, backbone="resnet")

    rclpy.spin(face_detection_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    face_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
