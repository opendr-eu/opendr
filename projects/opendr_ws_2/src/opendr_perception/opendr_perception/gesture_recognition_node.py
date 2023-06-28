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
import torch
from time import perf_counter

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2DArray
from opendr_bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.perception.gesture_recognition.gesture_recognition_learner import GestureRecognitionLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes


class GestureRecognitionNode(Node):

    def __init__(self, input_rgb_image_topic="image_raw", output_rgb_image_topic="/opendr/images",
                 detections_topic="/opendr/gestures", performance_topic=None, device="cuda",
                 model="plus_m_1.5x_416", threshold=0.5):
        """
        Creates a ROS2 Node for gesture recognition.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, no annotated
        image is published)
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the predictions (if None, no object detection message
        is published)
        :type detections_topic:  str
        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model: the name of the model of which we want to load the config file
        :type model: str
        """
        super().__init__('gesture_recognition_node')

        self.image_subscriber = self.create_subscription(ROS_Image, input_rgb_image_topic, self.callback, 1)

        if output_rgb_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_rgb_image_topic, 1)
        else:
            self.image_publisher = None

        if detections_topic is not None:
            self.object_publisher = self.create_publisher(Detection2DArray, detections_topic, 1)
        else:
            self.object_publisher = None

        if performance_topic is not None:
            self.performance_publisher = self.create_publisher(Float32, performance_topic, 1)
        else:
            self.performance_publisher = None

        self.bridge = ROS2Bridge()

        # Initialize the object detector
        self.gesture_model = GestureRecognitionLearner(model_to_use=model, device=device)
        self.gesture_model.download(path=".", verbose=True)
        self.gesture_model.load("./nanodet_{}".format(model))
        self.threshold = threshold
        self.get_logger().info("Gesture recognition node initialized.")

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        if self.performance_publisher:
            start_time = perf_counter()
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding='bgr8')

        # Run gesture recognition
        boxes = self.gesture_model.infer(image, conf_threshold=self.threshold)

        if self.performance_publisher:
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)  # NOQA
            fps_msg = Float32()
            fps_msg.data = fps
            self.performance_publisher.publish(fps_msg)

        # Publish gesture detections in ROS message
        if self.object_publisher is not None:
            self.object_publisher.publish(self.bridge.to_ros_boxes(boxes))

        if self.image_publisher is not None:
            # Get an OpenCV image back
            image = image.opencv()
            # Annotate image with gesture boxes
            image = draw_bounding_boxes(image, boxes, class_names=self.gesture_model.classes)
            # Convert the annotated OpenDR image to ROS2 image message using bridge and publish it
            self.image_publisher.publish(self.bridge.to_ros_image(Image(image), encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="image_raw")
    parser.add_argument("-o", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/rgb_gesture_images_annotated")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/rgb_gestures")
    parser.add_argument("--performance_topic", help="Topic name for performance messages, disabled (None) by default",
                        type=str, default=None)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", help="Model that config file will be used", type=str, default="plus_m_1.5x_416")
    parser.add_argument("--threshold", help="Confidence threshold for inference", default=0.5)
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

    gesture_recognition_node = GestureRecognitionNode(device=device, model=args.model,
                                                      input_rgb_image_topic=args.input_rgb_image_topic,
                                                      output_rgb_image_topic=args.output_rgb_image_topic,
                                                      detections_topic=args.detections_topic,
                                                      performance_topic=args.performance_topic,
                                                      threshold=args.threshold)

    rclpy.spin(gesture_recognition_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gesture_recognition_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
