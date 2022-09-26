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

import torch
import argparse
import cv2
import os
from opendr.engine.target import TrackingAnnotation
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image as ROS_Image
from opendr_ros2_bridge import ROS2Bridge
from opendr.engine.learners import Learner
from opendr.perception.object_tracking_2d import (
    ObjectTracking2DDeepSortLearner,
    ObjectTracking2DFairMotLearner
)
from opendr.engine.data import Image, ImageWithDetections


class ObjectTracking2DDeepSortNode(Node):
    def __init__(
        self,
        detector: Learner,
        input_image_topic="/usb_cam/image_raw",
        output_detection_topic="/opendr/detection",
        output_tracking_id_topic="/opendr/tracking_id",
        output_image_topic="/opendr/image_annotated",
        device="cuda:0",
        model_name="deep_sort",
        temp_dir="temp",
    ):
        """
        Creates a ROS2 Node for 2D object tracking
        :param detector: Learner to generate object detections
        :type detector: Learner
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param output_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_image_topic: str
        :param output_detection_topic: Topic to which we are publishing the detections
        :type output_detection_topic:  str
        :param output_tracking_id_topic: Topic to which we are publishing the tracking ids
        :type output_tracking_id_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model_name: the pretrained model to download or a saved model in temp_dir folder to use
        :type model_name: str
        :param temp_dir: the folder to download models
        :type temp_dir: str
        """

        super().__init__('object_tracking_2d_deep_sort_node')

        self.get_logger().info("Using model_name: {}".format(model_name))

        self.detector = detector
        self.learner = ObjectTracking2DDeepSortLearner(
            device=device, temp_path=temp_dir,
        )
        if not os.path.exists(os.path.join(temp_dir, model_name)):
            ObjectTracking2DDeepSortLearner.download(model_name, temp_dir)

        self.learner.load(os.path.join(temp_dir, model_name), verbose=True)

        # Initialize OpenDR ROSBridge object
        self.bridge = ROS2Bridge()
        self.tracking_id_publisher = self.create_publisher(
            Int32MultiArray, output_tracking_id_topic, 1
        )

        if output_image_topic is not None:
            self.output_image_publisher = self.create_publisher(
                ROS_Image, output_image_topic, 1
            )

        self.detection_publisher = self.create_publisher(
            Detection2DArray, output_detection_topic, 1
        )

        self.create_subscription(ROS_Image, input_image_topic, self.callback, 1)

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding="bgr8")
        detection_boxes = self.detector.infer(image)
        image_with_detections = ImageWithDetections(image.numpy(), detection_boxes)
        tracking_boxes = self.learner.infer(image_with_detections)

        if self.output_image_publisher is not None:
            frame = image.opencv()
            draw_predictions(frame, tracking_boxes)
            message = self.bridge.to_ros_image(
                Image(frame), encoding="bgr8"
            )
            self.output_image_publisher.publish(message)
            self.get_logger().info("Published annotated image")

        ids = [int(tracking_box.id) for tracking_box in tracking_boxes]

        # Convert detected boxes to ROS type and publish
        ros_boxes = self.bridge.to_ros_boxes(detection_boxes)
        if self.detection_publisher is not None:
            self.detection_publisher.publish(ros_boxes)
            self.get_logger().info("Published " + str(len(detection_boxes)) + " detection boxes")

        ros_ids = Int32MultiArray()
        ros_ids.data = ids

        if self.tracking_id_publisher is not None:
            self.tracking_id_publisher.publish(ros_ids)
            self.get_logger().info("Published " + str(len(ids)) + " tracking ids")


colors = [
    (255, 0, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (35, 69, 55),
    (43, 63, 54),
]


def draw_predictions(frame, predictions: TrackingAnnotation, is_centered=False, is_flipped_xy=True):
    global colors
    w, h, _ = frame.shape

    for prediction in predictions.boxes:
        prediction = prediction

        if not hasattr(prediction, "id"):
            prediction.id = 0

        color = colors[int(prediction.id) * 7 % len(colors)]

        x = prediction.left
        y = prediction.top

        if is_flipped_xy:
            x = prediction.top
            y = prediction.left

        if is_centered:
            x -= prediction.width
            y -= prediction.height

        cv2.rectangle(
            frame,
            (int(x), int(y)),
            (
                int(x + prediction.width),
                int(y + prediction.height),
            ),
            color,
            2,
        )


def main(
    args=None,
):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", help="Name of the trained model",
                        type=str, default="deep_sort")
    parser.add_argument("-t", "--temp_dir", help="Path to a temp dir with models",
                        type=str, default="temp")
    parser.add_argument("-i", "--input_image_topic",
                        help="Input Image topic provdied by either an image_dataset_node, webcam or any other image node",
                        type=str, default="/opendr/dataset_image")
    parser.add_argument("-od", "--output_detection_topic",
                        help="Output detections topic",
                        type=str, default="/opendr/detection")
    parser.add_argument("-ot", "--output_tracking_id_topic",
                        help="Output detections topic",
                        type=str, default="/opendr/tracking_id")
    parser.add_argument("-oi", "--output_image_topic",
                        help="Output detections topic",
                        type=str, default="/opendr/image_annotated")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
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

    detection_learner = ObjectTracking2DFairMotLearner(
        device=device, temp_path=args.temp_dir,
    )
    if not os.path.exists(os.path.join(args.temp_dir, "fairmot_dla34")):
        ObjectTracking2DFairMotLearner.download("fairmot_dla34", args.temp_dir)

    detection_learner.load(os.path.join(args.temp_dir, "fairmot_dla34"), verbose=True)

    deep_sort_node = ObjectTracking2DDeepSortNode(
        detector=detection_learner,
        device=device,
        model_name=args.model_name,
        input_image_topic=args.input_image_topic,
        temp_dir=args.temp_dir,
        output_detection_topic=args.output_detection_topic,
        output_tracking_id_topic=args.output_tracking_id_topic,
        output_image_topic=args.output_image_topic if args.output_image_topic != "None" else None,
    )
    rclpy.spin(deep_sort_node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    deep_sort_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
