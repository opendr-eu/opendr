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
import os
from opendr.engine.learners import Learner
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import PointCloud as ROS_PointCloud
from opendr_ros2_bridge import ROS2Bridge
from opendr.perception.object_tracking_3d import ObjectTracking3DAb3dmotLearner
from opendr.perception.object_detection_3d import VoxelObjectDetection3DLearner


class ObjectTracking3DAb3dmotNode(Node):
    def __init__(
        self,
        detector: Learner,
        input_point_cloud_topic="/opendr/dataset_point_cloud",
        output_detection3d_topic="/opendr/detection3d",
        output_tracking3d_id_topic="/opendr/tracking3d_id",
        device="cuda:0",
    ):
        """
        Creates a ROS2 Node for 3D object tracking
        :param detector: Learner that proides 3D object detections
        :type detector: Learner
        :param input_point_cloud_topic: Topic from which we are reading the input point cloud
        :type input_image_topic: str
        :param output_detection3d_topic: Topic to which we are publishing the annotations
        :type output_detection3d_topic:  str
        :param output_tracking3d_id_topic: Topic to which we are publishing the tracking ids
        :type output_tracking3d_id_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        super().__init__('object_tracking_3d_ab3dmot_node')

        self.detector = detector
        self.learner = ObjectTracking3DAb3dmotLearner(
            device=device
        )

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

        self.create_subscription(ROS_PointCloud, input_point_cloud_topic, self.callback, 1)

        self.get_logger().info("Ready to listen")

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        point_cloud = self.bridge.from_ros_point_cloud(data)
        detection_boxes = self.detector.infer(point_cloud)
        tracking_boxes = self.learner.infer(detection_boxes)

        # Convert detected boxes to ROS type and publish
        if self.detection_publisher is not None:
            ros_boxes = self.bridge.to_ros_boxes_3d(detection_boxes)
            self.detection_publisher.publish(ros_boxes)
            self.get_logger().info("Published " + str(len(detection_boxes)) + " detection boxes")

        if self.tracking_id_publisher is not None:
            ids = [tracking_box.id for tracking_box in tracking_boxes]
            ros_ids = Int32MultiArray()
            ros_ids.data = ids
            self.tracking_id_publisher.publish(ros_ids)
            self.get_logger().info("Published " + str(len(ids)) + " tracking ids")


def main(
    args=None,
):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--detector_model_name", help="Name of the trained model",
                        type=str, default="tanet_car_xyres_16")
    parser.add_argument(
        "-dc", "--detector_model_config_path", help="Path to a model .proto config",
        type=str, default=os.path.join(
            "..", "..", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "tanet",
            "car", "xyres_16.proto"
        )
    )
    parser.add_argument("-t", "--temp_dir", help="Path to a temp dir with models",
                        type=str, default="temp")
    parser.add_argument("-i", "--input_point_cloud_topic",
                        help="Point Cloud topic provided by either a point_cloud_dataset_node or any other 3D Point Cloud Node",
                        type=str, default="/opendr/dataset_point_cloud")
    parser.add_argument("-od", "--output_detection3d_topic",
                        help="Output detections topic",
                        type=str, default="/opendr/detection3d")
    parser.add_argument("-ot", "--output_tracking3d_id_topic",
                        help="Output tracking topic",
                        type=str, default="/opendr/tracking3d_id")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    input_point_cloud_topic = args.input_point_cloud_topic
    detector_model_name = args.detector_model_name
    temp_dir = args.temp_dir
    detector_model_config_path = args.detector_model_config_path
    output_detection3d_topic = args.output_detection3d_topic
    output_tracking3d_id_topic = args.output_tracking3d_id_topic

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

    detector = VoxelObjectDetection3DLearner(
        device=device,
        temp_path=temp_dir,
        model_config_path=detector_model_config_path
    )
    if not os.path.exists(os.path.join(temp_dir, detector_model_name)):
        VoxelObjectDetection3DLearner.download(detector_model_name, temp_dir)

    detector.load(os.path.join(temp_dir, detector_model_name), verbose=True)

    # created node object
    ab3dmot_node = ObjectTracking3DAb3dmotNode(
        detector=detector,
        device=device,
        input_point_cloud_topic=input_point_cloud_topic,
        output_detection3d_topic=output_detection3d_topic if output_detection3d_topic != "None" else None,
        output_tracking3d_id_topic=output_tracking3d_id_topic if output_tracking3d_id_topic != "None" else None,
    )

    rclpy.spin(ab3dmot_node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ab3dmot_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
