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
from opendr_ros2_bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.perception.object_detection_2d import DetrLearner
from opendr.perception.object_detection_2d.detr.algorithm.util.draw import draw


class ObjectDetectionDetrNode(Node):

    def __init__(self, input_image_topic="image_raw", output_image_topic="/opendr/image_boxes_annotated",
                 detections_topic="/opendr/objects", device="cuda"):
        super().__init__('object_detection_detr_node')

        if output_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_image_topic, 1)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.bbox_publisher = self.create_publisher(Detection2DArray, detections_topic, 1)
        else:
            self.bbox_publisher = None

        self.image_subscriber = self.create_subscription(ROS_Image, input_image_topic, self.callback, 1)

        self.bridge = ROS2Bridge()

        self.detr_learner = DetrLearner(device=device)
        self.detr_learner.download(path=".", verbose=True)

    def callback(self, data):
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        cv2.imshow("image", image.opencv())
        cv2.waitKey(5)

        boxes = self.detr_learner.infer(image)

        image = np.float32(image.opencv())

        # Convert detected boxes to ROS type and publish
        ros_boxes = self.bridge.to_ros_bounding_box_list(boxes)
        if self.bbox_publisher is not None:
            self.bbox_publisher.publish(ros_boxes)

        # Annotate image and publish result
        image = draw(image, boxes)
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

    object_detection_detr_node = ObjectDetectionDetrNode(device=device)

    rclpy.spin(object_detection_detr_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detection_detr_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
