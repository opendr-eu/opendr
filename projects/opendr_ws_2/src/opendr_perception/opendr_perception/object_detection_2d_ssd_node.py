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

from opendr.engine.data import Image
from opendr.perception.object_detection_2d import SingleShotDetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.perception.object_detection_2d import Seq2SeqNMSLearner, SoftNMS, FastNMS, ClusterNMS


class ObjectDetectionSSDNode(Node):

    def __init__(self, input_rgb_image_topic="image_raw", output_rgb_image_topic="/opendr/image_objects_annotated",
                 detections_topic="/opendr/objects", device="cuda", backbone="vgg16_atrous", nms_type='default'):
        """
        Creates a ROS2 Node for object detection with SSD.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, no object detection message
        is published)
        :type detections_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param backbone: backbone network
        :type backbone: str
        :param nms_type: type of NMS method, can be one
        of 'default', 'seq2seq-nms', 'soft-nms', 'fast-nms', 'cluster-nms'
        :type nms_type: str
        """
        super().__init__('opendr_object_detection_2d_ssd_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_rgb_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.object_publisher = self.create_publisher(Detection2DArray, detections_topic, 1)
        else:
            self.object_publisher = None

        self.bridge = ROS2Bridge()

        self.object_detector = SingleShotDetectorLearner(backbone=backbone, device=device)
        self.object_detector.download(path=".", verbose=True)
        self.object_detector.load("ssd_default_person")
        self.custom_nms = None

        # Initialize NMS if selected
        if nms_type == 'seq2seq-nms':
            self.custom_nms = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering=0.8,
                                                app_feats='fmod', device=device)
            self.custom_nms.download(model_name='seq2seq_pets_jpd_fmod', path='.')
            self.custom_nms.load('./seq2seq_pets_jpd_fmod/', verbose=True)
            self.get_logger().info("Object Detection 2D SSD node seq2seq-nms initialized.")
        elif nms_type == 'soft-nms':
            self.custom_nms = SoftNMS(nms_thres=0.45, device=device)
            self.get_logger().info("Object Detection 2D SSD node soft-nms initialized.")
        elif nms_type == 'fast-nms':
            self.custom_nms = FastNMS(device=device)
            self.get_logger().info("Object Detection 2D SSD node fast-nms initialized.")
        elif nms_type == 'cluster-nms':
            self.custom_nms = ClusterNMS(device=device)
            self.get_logger().info("Object Detection 2D SSD node cluster-nms initialized.")
        else:
            self.get_logger().info("Object Detection 2D SSD node using default NMS.")

        self.get_logger().info("Object Detection 2D SSD node initialized.")

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run object detection
        boxes = self.object_detector.infer(image, threshold=0.45, keep_size=False, custom_nms=self.custom_nms)

        if self.object_publisher is not None:
            # Publish detections in ROS message
            ros_boxes = self.bridge.to_ros_boxes(boxes)  # Convert to ROS boxes
            self.object_publisher.publish(ros_boxes)

        if self.image_publisher is not None:
            # Get an OpenCV image back
            image = image.opencv()
            # Annotate image with object detection boxes
            image = draw_bounding_boxes(image, boxes, class_names=self.object_detector.classes)
            # Convert the annotated OpenDR image to ROS2 image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/image_objects_annotated")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=lambda value: value if value.lower() != "none" else None, default="/opendr/objects")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backbone", help="Backbone network, defaults to vgg16_atrous",
                        type=str, default="vgg16_atrous", choices=["vgg16_atrous"])
    parser.add_argument("--nms_type", help="Non-Maximum Suppression type, defaults to \"default\", options are "
                                           "\"seq2seq-nms\", \"soft-nms\", \"fast-nms\", \"cluster-nms\"",
                        type=str, default="default",
                        choices=["default", "seq2seq-nms", "soft-nms", "fast-nms", "cluster-nms"])
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

    object_detection_ssd_node = ObjectDetectionSSDNode(device=device, backbone=args.backbone, nms_type=args.nms_type,
                                                       input_rgb_image_topic=args.input_rgb_image_topic,
                                                       output_rgb_image_topic=args.output_rgb_image_topic,
                                                       detections_topic=args.detections_topic)

    rclpy.spin(object_detection_ssd_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detection_ssd_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
