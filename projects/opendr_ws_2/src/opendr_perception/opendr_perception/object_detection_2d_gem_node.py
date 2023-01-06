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
import cv2
import message_filters
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from opendr_bridge import ROS2Bridge
from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2DArray

from opendr.engine.data import Image
from opendr.perception.object_detection_2d import GemLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes


class ObjectDetectionGemNode(Node):
    def __init__(
        self,
        input_rgb_image_topic="/camera/color/image_raw",
        input_infra_image_topic="/camera/infra/image_raw",
        output_rgb_image_topic="/opendr/rgb_image_objects_annotated",
        output_infra_image_topic="/opendr/infra_image_objects_annotated",
        detections_topic="/opendr/objects",
        device="cuda",
        pts_rgb=None,
        pts_infra=None,
    ):
        """
        Creates a ROS2 Node for object detection with GEM
        :param input_rgb_image_topic: Topic from which we are reading the input rgb image
        :type input_rgb_image_topic: str
        :param input_infra_image_topic: Topic from which we are reading the input infrared image
        :type: input_infra_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated rgb image (if None, we are not
        publishing annotated image)
        :type output_rgb_image_topic: str
        :param output_infra_image_topic: Topic to which we are publishing the annotated infrared image (if None, we are not
        publishing annotated image)
        :type output_infra_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, we are
        not publishing annotations)
        :type detections_topic:  str
        :param device: Device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param pts_rgb: Point on the rgb image that define alignment with the infrared image. These are camera
        specific and can be obtained using get_color_infra_alignment.py which is located in the
        opendr/perception/object_detection2d/utils module.
        :type pts_rgb: {list, numpy.ndarray}
        :param pts_infra: Points on the infrared image that define alignment with rgb image. These are camera specific
        and can be obtained using get_color_infra_alignment.py which is located in the
        opendr/perception/object_detection2d/utils module.
        :type pts_infra: {list, numpy.ndarray}
        """
        super().__init__("opendr_object_detection_2d_gem_node")

        if output_rgb_image_topic is not None:
            self.rgb_publisher = self.create_publisher(msg_type=ROS_Image, topic=output_rgb_image_topic, qos_profile=10)
        else:
            self.rgb_publisher = None
        if output_infra_image_topic is not None:
            self.ir_publisher = self.create_publisher(msg_type=ROS_Image, topic=output_infra_image_topic, qos_profile=10)
        else:
            self.ir_publisher = None

        if detections_topic is not None:
            self.detection_publisher = self.create_publisher(msg_type=Detection2DArray, topic=detections_topic, qos_profile=10)
        else:
            self.detection_publisher = None
        if pts_infra is None:
            pts_infra = np.array(
                [
                    [478, 248],
                    [465, 338],
                    [458, 325],
                    [468, 256],
                    [341, 240],
                    [335, 310],
                    [324, 321],
                    [311, 383],
                    [434, 365],
                    [135, 384],
                    [67, 257],
                    [167, 206],
                    [124, 131],
                    [364, 276],
                    [424, 269],
                    [277, 131],
                    [41, 310],
                    [202, 320],
                    [188, 318],
                    [188, 308],
                    [196, 241],
                    [499, 317],
                    [311, 164],
                    [220, 216],
                    [435, 352],
                    [213, 363],
                    [390, 364],
                    [212, 368],
                    [390, 370],
                    [467, 324],
                    [415, 364],
                ]
            )
            self.get_logger().warn(
                "\nUsing default calibration values for pts_infra!" +
                "\nThese are probably incorrect." +
                "\nThe correct values for pts_infra can be found by running get_rgb_infra_alignment.py." +
                "\nThis file is located in the opendr/perception/object_detection2d/utils module."
            )
        if pts_rgb is None:
            pts_rgb = np.array(
                [
                    [910, 397],
                    [889, 572],
                    [874, 552],
                    [891, 411],
                    [635, 385],
                    [619, 525],
                    [603, 544],
                    [576, 682],
                    [810, 619],
                    [216, 688],
                    [90, 423],
                    [281, 310],
                    [193, 163],
                    [684, 449],
                    [806, 431],
                    [504, 170],
                    [24, 538],
                    [353, 552],
                    [323, 550],
                    [323, 529],
                    [344, 387],
                    [961, 533],
                    [570, 233],
                    [392, 336],
                    [831, 610],
                    [378, 638],
                    [742, 630],
                    [378, 648],
                    [742, 640],
                    [895, 550],
                    [787, 630],
                ]
            )
            self.get_logger().warn(
                "\nUsing default calibration values for pts_rgb!" +
                "\nThese are probably incorrect." +
                "\nThe correct values for pts_rgb can be found by running get_color_infra_alignment.py." +
                "\nThis file is located in the opendr/perception/object_detection2d/utils module."
            )
        # Object classes
        self.classes = ["N/A", "chair", "cycle", "bin", "laptop", "drill", "rocker"]

        # Estimating Homography matrix for aligning infra with rgb
        self.h, status = cv2.findHomography(pts_infra, pts_rgb)

        self.bridge = ROS2Bridge()

        # Initialize the detection estimation
        model_backbone = "resnet50"

        self.gem_learner = GemLearner(
            backbone=model_backbone,
            num_classes=7,
            device=device,
        )
        self.gem_learner.fusion_method = "sc_avg"
        self.gem_learner.download(path=".", verbose=True)

        # Subscribers
        msg_rgb = message_filters.Subscriber(self, ROS_Image, input_rgb_image_topic, 1)
        msg_ir = message_filters.Subscriber(self, ROS_Image, input_infra_image_topic, 1)

        sync = message_filters.TimeSynchronizer([msg_rgb, msg_ir], 1)
        sync.registerCallback(self.callback)

    def callback(self, msg_rgb, msg_ir):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param msg_rgb: input rgb image message
        :type msg_rgb: sensor_msgs.msg.Image
        :param msg_ir: input infrared image message
        :type msg_ir: sensor_msgs.msg.Image
        """
        # Convert images to OpenDR standard
        image_rgb = self.bridge.from_ros_image(msg_rgb).opencv()
        image_ir_raw = self.bridge.from_ros_image(msg_ir, "bgr8").opencv()
        image_ir = cv2.warpPerspective(image_ir_raw, self.h, (image_rgb.shape[1], image_rgb.shape[0]))

        # Perform inference on images
        boxes, w_sensor1, _ = self.gem_learner.infer(image_rgb, image_ir)

        #  Annotate image and publish results:
        if self.detection_publisher is not None:
            ros_detection = self.bridge.to_ros_bounding_box_list(boxes)
            self.detection_publisher.publish(ros_detection)
            # We can get the data back using self.bridge.from_ros_bounding_box_list(ros_detection)
            # e.g., opendr_detection = self.bridge.from_ros_bounding_box_list(ros_detection)

        if self.rgb_publisher is not None:
            plot_rgb = draw_bounding_boxes(image_rgb, boxes, class_names=self.classes)
            message = self.bridge.to_ros_image(Image(np.uint8(plot_rgb)))
            self.rgb_publisher.publish(message)
        if self.ir_publisher is not None:
            plot_ir = draw_bounding_boxes(image_ir, boxes, class_names=self.classes)
            message = self.bridge.to_ros_image(Image(np.uint8(plot_ir)))
            self.ir_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/camera/color/image_raw")
    parser.add_argument("-ii", "--input_infra_image_topic", help="Topic name for input infrared image",
                        type=str, default="/camera/infra/image_raw")
    parser.add_argument("-oc", "--output_rgb_image_topic", help="Topic name for output annotated rgb image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/rgb_image_objects_annotated")
    parser.add_argument("-oi", "--output_infra_image_topic", help="Topic name for output annotated infrared image",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/infra_image_objects_annotated")
    parser.add_argument("-d", "--detections_topic", help="Topic name for detection messages",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/objects")
    parser.add_argument("--device", help='Device to use, either "cpu" or "cuda", defaults to "cuda"',
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

    gem_node = ObjectDetectionGemNode(
        device=device,
        input_rgb_image_topic=args.input_rgb_image_topic,
        output_rgb_image_topic=args.output_rgb_image_topic,
        input_infra_image_topic=args.input_infra_image_topic,
        output_infra_image_topic=args.output_infra_image_topic,
        detections_topic=args.detections_topic,
    )

    rclpy.spin(gem_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gem_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
