#!/usr/bin/env python
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

import torch
import argparse
import os
from time import perf_counter
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import Int32MultiArray, Float32
from sensor_msgs.msg import PointCloud as ROS_PointCloud
from opendr_bridge import ROS2Bridge
from opendr.perception.object_tracking_3d import ObjectTracking3DVpitLearner
from opendr.engine.target import TrackingAnnotation3D

config_root = "./src/opendr/perception/object_tracking_3d/single_object_tracking/vpit/second_detector/configs"

config_tanet_car = os.path.join(config_root, "tanet/car/xyres_16.proto")
config_pointpillars_car = os.path.join(config_root, "pointpillars/car/xyres_16.proto")
config_pointpillars_car_tracking = os.path.join(config_root, "pointpillars/car/xyres_16_tracking.proto")
config_pointpillars_car_tracking_s = os.path.join(config_root, "pointpillars/car/xyres_16_tracking_s.proto")
config_tanet_car_tracking = os.path.join(config_root, "tanet/car/xyres_16_tracking.proto")
config_tanet_car_tracking_s = os.path.join(config_root, "tanet/car/xyres_16_tracking_s.proto")

backbone_configs = {
    "pp": config_pointpillars_car,
    "spp": config_pointpillars_car_tracking,
    "spps": config_pointpillars_car_tracking_s,
    "tanet": config_tanet_car,
    "stanet": config_tanet_car_tracking,
    "stanets": config_tanet_car_tracking_s,
}


class ObjectTracking3DVpitNode(Node):
    def __init__(
        self,
        backbone="pp",
        model_name=None,
        input_point_cloud_topic="/opendr/dataset_point_cloud",
        input_detection3d_topic="/opendr/dataset_detection3d",
        output_detection3d_topic="/opendr/detection3d",
        output_tracking3d_id_topic="/opendr/tracking3d_id",
        performance_topic=None,
        device="cuda:0",
    ):
        """
        Creates a ROS2 Node for 3D object tracking
        :param detector: Learner that provides 3D object detections
        :type detector: Learner
        :param input_point_cloud_topic: Topic from which we are reading the input point cloud
        :type input_point_cloud_topic: str
        :param output_detection3d_topic: Topic to which we are publishing the annotations
        :type output_detection3d_topic:  str
        :param output_tracking3d_id_topic: Topic to which we are publishing the tracking ids
        :type output_tracking3d_id_topic:  str
        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        super().__init__("opendr_object_tracking_3d_vpit_node")

        self.learner = ObjectTracking3DVpitLearner(
            model_config_path=backbone_configs[backbone],
            device=device,
            backbone=backbone,
            extrapolation_mode="linear+",
            window_influence=0.85,
            score_upscale=8,
            rotation_penalty=0.98,
            target_feature_merge_scale=0.0,
            min_top_score=None,
            offset_interpolation=0.25,
            search_type="small",
            target_type="normal",
            feature_blocks=1,
            target_size=[-1, -1],
            search_size=[-1, -1],
            context_amount=0.25,
            lr=0.0001,
            r_pos=2,
            augment=None,
        )

        if model_name is not None and not os.path.exists(
            "./" + model_name
        ):
            self.learner.download(model_name, "./")
        self.learner.load("./" + model_name, full=True, verbose=True)
        print("Learner created")
        # Initialize OpenDR ROSBridge object
        self.bridge = ROS2Bridge()

        if output_detection3d_topic is not None:
            self.detection_publisher = self.create_publisher(
                Detection3DArray, output_detection3d_topic, 1
            )

        if output_tracking3d_id_topic is not None:
            self.tracking_id_publisher = self.create_publisher(
                Int32MultiArray, output_tracking3d_id_topic, 1
            )

        if performance_topic is not None:
            self.performance_publisher = self.create_subscription(
                Float32, performance_topic, 1
            )
        else:
            self.performance_publisher = None

        self.create_subscription(
            ROS_PointCloud, input_point_cloud_topic, self.callback_pc, 1
        )
        self.create_subscription(
            Detection3DArray, input_detection3d_topic, self.callback_det, 1
        )

        self.get_logger().info("Object Tracking 3D Vpit Node initialized.")
        self.last_point_cloud = None
        self.last_input_detection = None
        self.waiting_for_init = True
        self.frame = 0

    def init(self):

        box = self.last_input_detection
        label = TrackingAnnotation3D(
            box.name,
            box.truncated,
            box.occluded,
            box.alpha,
            box.bbox2d,
            box.dimensions,
            box.location,
            box.rotation_y,
            0,
            box.confidence,
            self.frame,
        )

        self.learner.init(self.last_point_cloud, label)
        self.waiting_for_init = False

    def callback_pc(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        if self.performance_publisher:
            start_time = perf_counter()
        # Convert sensor_msgs.msg.Image into OpenDR Image
        point_cloud = self.bridge.from_ros_point_cloud(data)
        self.last_point_cloud = point_cloud

        if self.waiting_for_init:
            if self.last_input_detection is not None:
                self.init()
            else:
                return

        tracking_boxes = self.learner.infer(point_cloud)

        if self.performance_publisher:
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)  # NOQA
            fps_msg = Float32()
            fps_msg.data = fps
            self.performance_publisher.publish(fps_msg)

        if self.detection_publisher is not None:
            # Convert detected boxes to ROS type and publish
            detection_boxes = tracking_boxes.bounding_box_3d_list()
            self.detection_publisher.publish(
                self.bridge.to_ros_boxes_3d(detection_boxes)
            )
            self.get_logger().info(
                "Published " + str(len(detection_boxes)) + " detection boxes"
            )

        if self.tracking_id_publisher is not None:
            ids = [tracking_box.id for tracking_box in tracking_boxes]
            ros_ids = Int32MultiArray()
            ros_ids.data = ids
            self.tracking_id_publisher.publish(ros_ids)
            self.get_logger().info("Published " + str(len(ids)) + " tracking ids")

        self.frame += 1

    def callback_det(self, data):
        """
        Callback that processes the initial detection data.
        :param data: input message
        :type data: vision_msgs.msg.Detection3DArray
        """

        self.last_input_detection = self.bridge.from_ros_boxes_3d(data)[0]

        if self.last_point_cloud is not None:
            self.init()


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ipc",
        "--input_point_cloud_topic",
        help="Point Cloud topic provided by either a point_cloud_dataset_node or any other 3D Point Cloud Node",
        type=str,
        default="/opendr/dataset_point_cloud",
    )
    parser.add_argument(
        "-idet",
        "--input_detection3d_topic",
        help="Detection 3d topic provided by either a 3D detector or a dataset",
        type=str,
        default="/opendr/dataset_detection3d",
    )
    parser.add_argument(
        "-d",
        "--detections_topic",
        help="Output detections topic",
        type=lambda value: value if value.lower() != "none" else None,
        default="/opendr/objects3d",
    )
    parser.add_argument(
        "-t",
        "--tracking3d_id_topic",
        help="Output tracking ids topic with the same element count as in output_detection_topic",
        type=lambda value: value if value.lower() != "none" else None,
        default="/opendr/objects_tracking_id",
    )
    parser.add_argument(
        "--performance_topic",
        help="Topic name for performance messages, disabled (None) by default",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        help='Device to use, either "cpu" or "cuda", defaults to "cuda"',
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "-bb",
        "--backbone",
        help="Name of the backbone model",
        type=str,
        default="pp",
        choices=["pp, spp, spps, tanet, stanet, stanets"],
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        help="Name of the trained model to load",
        type=lambda value: value if value.lower() != "none" else None,
        default="vpit",
    )

    args = parser.parse_args()

    input_point_cloud_topic = args.input_point_cloud_topic
    input_detection3d_topic = args.input_detection3d_topic
    output_detection3d_topic = args.detections_topic
    output_tracking3d_id_topic = args.tracking3d_id_topic
    backbone = args.backbone
    model_name = args.model_name

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

    vpit_node = ObjectTracking3DVpitNode(
        backbone=backbone,
        model_name=model_name,
        device=device,
        input_point_cloud_topic=input_point_cloud_topic,
        input_detection3d_topic=input_detection3d_topic,
        output_detection3d_topic=output_detection3d_topic,
        output_tracking3d_id_topic=output_tracking3d_id_topic,
        performance_topic=args.performance_topic,
    )

    rclpy.spin(vpit_node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vpit_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
