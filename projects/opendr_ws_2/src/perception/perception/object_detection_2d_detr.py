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
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2DArray
from ros2_bridge.bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.perception.object_detection_2d import DetrLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes


class DetrNode(Node):
    def __init__(
        self,
        input_image_topic="/image_raw",
        output_image_topic="/opendr/image_boxes_annotated",
        detection_annotations_topic="/opendr/objects",
        device="cuda",
    ):
        """
        Creates a ROS2 Node for object detection with DETR
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param output_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_image_topic: str
        :param detection_annotations_topic: Topic to which we are publishing the annotations (if None, we are not publishing
        annotations)
        :type detection_annotations_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        super().__init__("detr_detection_node")

        if output_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_image_topic, 10)
        else:
            self.image_publisher = None

        if detection_annotations_topic is not None:
            self.detection_publisher = self.create_publisher(Detection2DArray, detection_annotations_topic, 10)
        else:
            self.detection_publisher = None

        self.image_subscriber = self.create_subscription(ROS_Image, input_image_topic, self.callback, 10)

        self.bridge = ROS2Bridge()

        self.class_names = [
            "N/A",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "N/A",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "N/A",
            "backpack",
            "umbrella",
            "N/A",
            "N/A",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "N/A",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "N/A",
            "dining table",
            "N/A",
            "N/A",
            "toilet",
            "N/A",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "N/A",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        # Initialize the detection estimation
        self.detr_learner = DetrLearner(device=device)
        self.detr_learner.download(path=".", verbose=True)

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.from_ros_image(data, encoding="bgr8")

        # Run detection estimation
        boxes = self.detr_learner.infer(image)

        # Get an OpenCV image back
        image = np.float32(image.opencv())

        #  Annotate image and publish results:
        if self.detection_publisher is not None:
            ros_detection = self.bridge.to_ros_bounding_box_list(boxes)
            self.detection_publisher.publish(ros_detection)
            # We get can the data back using self.bridge.from_ros_bounding_box_list(ros_detection)
            # e.g., opendr_detection = self.bridge.from_ros_bounding_box_list(ros_detection)

        if self.image_publisher is not None:
            image = draw_bounding_boxes(image, boxes, class_names=self.class_names)
            message = self.bridge.to_ros_image(Image(image), encoding="bgr8")
            self.image_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    try:
        if torch.cuda.is_available():
            print("GPU found.")
            device = "cuda"
        else:
            print("GPU not found. Using CPU instead.")
            device = "cpu"
    except:
        device = "cpu"

    detr_node = DetrNode(device=device)

    rclpy.spin(detr_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detr_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
