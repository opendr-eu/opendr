#!/usr/bin/env python3
# Copyright 2020-2024 OpenDR European Project
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
from time import perf_counter

import numpy as np
import rospy
import torch
from opendr_bridge import ROSBridge
from sensor_msgs.msg import Image as ROS_Image
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2DArray

from opendr.engine.data import Image
from opendr.engine.target import Heatmap
from opendr.perception.semantic_segmentation import YOLOv8SegLearner


class SemanticSegmantationYOLOV8Node:

    def __init__(self, input_rgb_image_topic="/usb_cam/image_raw", output_heatmap_topic="/opendr/heatmap",
                 output_rgb_image_topic="/opendr/heatmap_visualization", detections_topic="/opendr/objects",
                 performance_topic=None, device="cuda", model_name="yolov8s-seg"):
        """
        Creates a ROS Node for semantic segmentation with YOLOv8.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_heatmap_topic: Topic to which we are publishing the heatmap in the form of a ROS image containing
        class ids
        :type output_heatmap_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the heatmap image blended with the
        input image and a class legend for visualization purposes
        :type output_rgb_image_topic: str
        :param detections_topic: Topic to which we are publishing the bounding boxes (if None, no object detection
        message is published)
        :type detections_topic:  str
        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model_name: network architecture name
        :type model_name: str
        """
        self.input_rgb_image_topic = input_rgb_image_topic

        if output_heatmap_topic is not None:
            self.heatmap_publisher = rospy.Publisher(output_heatmap_topic, ROS_Image, queue_size=1)
        else:
            self.heatmap_publisher = None

        if output_rgb_image_topic is not None:
            self.visualization_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=1)
        else:
            self.visualization_publisher = None

        if detections_topic is not None:
            self.object_publisher = rospy.Publisher(detections_topic, Detection2DArray, queue_size=1)
        else:
            self.object_publisher = None

        if performance_topic is not None:
            self.performance_publisher = rospy.Publisher(performance_topic, Float32, queue_size=1)
        else:
            self.performance_publisher = None

        self.bridge = ROSBridge()

        # Initialize the learner
        self.learner = YOLOv8SegLearner(model_name=model_name, device=device)

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_semantic_segmentation_yolov8_node', anonymous=True)
        rospy.Subscriber(self.input_rgb_image_topic, ROS_Image, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Semantic segmentation YOLOv8 node started.")
        rospy.spin()

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

        # Run semantic segmentation
        heatmap = self.learner.infer(image)

        if self.performance_publisher:
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)  # NOQA
            fps_msg = Float32()
            fps_msg.data = fps
            self.performance_publisher.publish(fps_msg)

        # Publish heatmap in the form of an image containing class ids
        if self.heatmap_publisher is not None:
            heatmap = Heatmap(heatmap.data.astype(np.uint8))  # Convert to uint8
            self.heatmap_publisher.publish(self.bridge.to_ros_image(heatmap))

        # Publish bounding box detections in ROS message
        if self.object_publisher is not None:
            boxes = self.learner.get_bboxes()
            self.object_publisher.publish(self.bridge.to_ros_bounding_box_list(boxes))

        # Publish heatmap color visualization
        if self.visualization_publisher is not None:
            img_vis = self.learner.get_visualization(labels=True, boxes=self.object_publisher is not None,
                                                     masks=True, conf=True)
            self.visualization_publisher.publish(self.bridge.to_ros_image(Image(img_vis), encoding='bgr8'))  # NOQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_heatmap_topic", help="Topic to which we are publishing the heatmap in the form "
                                                             "of a ROS image containing class ids",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/heatmap")
    parser.add_argument("-ov", "--output_rgb_image_topic", help="Topic to which we are publishing the heatmap image "
                                                                "blended with the input image and a class legend for "
                                                                "visualization purposes",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/heatmap_visualization")
    parser.add_argument("-d", "--detections_topic", help="Topic name for object detection messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/objects")
    parser.add_argument("--performance_topic", help="Topic name for performance messages, disabled (None) by default",
                        type=str, default=None)
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model_name", help="Network architecture, defaults to \"yolov8s-seg\"",
                        type=str, default="yolov8s-seg", choices=["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg",
                                                                  "yolov8x-seg", "custom"])
    args = parser.parse_args(rospy.myargv()[1:])

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

    semantic_segmentation_yolov8_node = SemanticSegmantationYOLOV8Node(device=device, model_name=args.model_name,
                                                                       input_rgb_image_topic=args.input_rgb_image_topic,
                                                                       output_heatmap_topic=args.output_heatmap_topic,
                                                                       output_rgb_image_topic=args.output_rgb_image_topic,
                                                                       detections_topic=args.detections_topic,
                                                                       performance_topic=args.performance_topic)
    semantic_segmentation_yolov8_node.listen()


if __name__ == '__main__':
    main()
