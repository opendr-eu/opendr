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

from opendr.engine.data import Image
from opendr.perception.object_detection_2d import SingleShotDetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.perception.object_detection_2d import Seq2SeqNMSLearner, SoftNMS, FastNMS, ClusterNMS


class ObjectDetectionSSDNode(Node):

    def __init__(self, input_image_topic="image_raw", output_image_topic="/opendr/image_boxes_annotated",
                 detections_topic="/opendr/objects", device="cuda", backbone="vgg16_atrous", nms_type='default'):
        super().__init__('object_detection_ssd_node')

        if output_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_image_topic, 10)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.bbox_publisher = self.create_publisher(Detection2DArray, detections_topic, 10)
        else:
            self.bbox_publisher = None

        self.image_subscriber = self.create_subscription(ROS_Image, input_image_topic, self.callback, 10)

        self.bridge = ROS2Bridge()

        self.object_detector = SingleShotDetectorLearner(backbone=backbone, device=device)
        self.object_detector.download(path=".", verbose=True)
        self.object_detector.load("ssd_default_person")
        self.class_names = self.object_detector.classes
        self.custom_nms = None

        # Initialize Seq2Seq-NMS if selected
        if nms_type == 'seq2seq-nms':
            self.custom_nms = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering=0.8,
                                                app_feats='fmod', device=self.device)
            self.custom_nms.download(model_name='seq2seq_pets_jpd', path='.')
            self.custom_nms.load('./seq2seq_pets_jpd/', verbose=True)
        elif nms_type == 'soft-nms':
            self.custom_nms = SoftNMS(nms_thres=0.45, device=self.device)
        elif nms_type == 'fast-nms':
            self.custom_nms = FastNMS(nms_thres=0.45, device=self.device)
        elif nms_type == 'cluster-nms':
            self.custom_nms = ClusterNMS(nms_thres=0.45, device=self.device)

    def callback(self, data):
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        cv2.imshow("image", image.opencv())
        cv2.waitKey(5)

        boxes = self.object_detector.infer(image, threshold=0.45, keep_size=False, custom_nms=self.custom_nms)

        image = np.float32(image.opencv())

        # Convert detected boxes to ROS type and publish
        ros_boxes = self.bridge.to_ros_boxes(boxes)
        if self.bbox_publisher is not None:
            self.bbox_publisher.publish(ros_boxes)

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

    object_detection_ssd_node = ObjectDetectionSSDNode(device=device)

    rclpy.spin(object_detection_ssd_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detection_ssd_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
