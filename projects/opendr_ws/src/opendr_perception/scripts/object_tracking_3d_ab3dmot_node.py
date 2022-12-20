#!/usr/bin/env python3
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

import argparse
import os
import torch
import rospy
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import PointCloud as ROS_PointCloud
from opendr_bridge import ROSBridge
from opendr.perception.object_tracking_3d import ObjectTracking3DAb3dmotLearner
from opendr.perception.object_detection_3d import VoxelObjectDetection3DLearner


class ObjectTracking3DAb3dmotNode:
    def __init__(
        self,
        detector=None,
        input_point_cloud_topic="/opendr/dataset_point_cloud",
        output_detection3d_topic="/opendr/detection3d",
        output_tracking3d_id_topic="/opendr/tracking3d_id",
        device="cuda:0",
    ):
        """
        Creates a ROS Node for 3D object tracking
        :param detector: Learner that provides 3D object detections
        :type detector: Learner
        :param input_point_cloud_topic: Topic from which we are reading the input point cloud
        :type input_point_cloud_topic: str
        :param output_detection3d_topic: Topic to which we are publishing the annotations
        :type output_detection3d_topic:  str
        :param output_tracking3d_id_topic: Topic to which we are publishing the tracking ids
        :type output_tracking3d_id_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """

        self.detector = detector
        self.learner = ObjectTracking3DAb3dmotLearner(
            device=device
        )

        self.bridge = ROSBridge()
        self.input_point_cloud_topic = input_point_cloud_topic

        if output_detection3d_topic is not None:
            self.detection_publisher = rospy.Publisher(
                output_detection3d_topic, Detection3DArray, queue_size=10
            )

        if output_tracking3d_id_topic is not None:
            self.tracking_id_publisher = rospy.Publisher(
                output_tracking3d_id_topic, Int32MultiArray, queue_size=10
            )

        rospy.Subscriber(input_point_cloud_topic, ROS_PointCloud, self.callback)

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

        if self.detection_publisher is not None:
            # Convert detected boxes to ROS type and publish
            ros_boxes = self.bridge.to_ros_boxes_3d(detection_boxes, classes=["Car", "Van", "Truck", "Pedestrian", "Cyclist"])
            self.detection_publisher.publish(ros_boxes)

        if self.tracking_id_publisher is not None:
            ids = [tracking_box.id for tracking_box in tracking_boxes]
            ros_ids = Int32MultiArray()
            ros_ids.data = ids
            self.tracking_id_publisher.publish(ros_ids)

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_object_ab3dmot_tracking_3d_node', anonymous=True)
        rospy.Subscriber(self.input_point_cloud_topic, ROS_PointCloud, self.callback, queue_size=1, buff_size=10000000)

        rospy.loginfo("Object Tracking 3D Ab3dmot Node started.")
        rospy.spin()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_point_cloud_topic",
                        help="Point Cloud topic provided by either a point_cloud_dataset_node or any other 3D Point Cloud Node",
                        type=str, default="/opendr/dataset_point_cloud")
    parser.add_argument("-d", "--detections_topic",
                        help="Output detections topic",
                        type=lambda value: value if value.lower() != "none" else None, default="/opendr/objects3d")
    parser.add_argument("-t", "--tracking3d_id_topic",
                        help="Output tracking ids topic with the same element count as in output_detection_topic",
                        type=lambda value: value if value.lower() != "none" else None, default="/opendr/objects_tracking_id")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("-dn", "--detector_model_name", help="Name of the trained model",
                        type=str, default="tanet_car_xyres_16", choices=["tanet_car_xyres_16"])
    parser.add_argument(
        "-dc", "--detector_model_config_path", help="Path to a model .proto config",
        type=str, default=os.path.join(
            "$OPENDR_HOME", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "tanet",
            "car", "xyres_16.proto"
        )
    )
    parser.add_argument("-td", "--temp_dir", help="Path to a temporary directory with models",
                        type=str, default="temp")
    args = parser.parse_args()

    input_point_cloud_topic = args.input_point_cloud_topic
    detector_model_name = args.detector_model_name
    temp_dir = args.temp_dir
    detector_model_config_path = args.detector_model_config_path
    output_detection3d_topic = args.detections_topic
    output_tracking3d_id_topic = args.tracking3d_id_topic

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

    ab3dmot_node = ObjectTracking3DAb3dmotNode(
        detector=detector,
        device=device,
        input_point_cloud_topic=input_point_cloud_topic,
        output_detection3d_topic=output_detection3d_topic,
        output_tracking3d_id_topic=output_tracking3d_id_topic,
    )

    ab3dmot_node.listen()


if __name__ == '__main__':
    main()
