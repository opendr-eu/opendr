#!/usr/bin/env python3
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
import os
import rospy
from vision_msgs.msg import Detection3DArray
from sensor_msgs.msg import PointCloud as ROS_PointCloud
from opendr_bridge import ROSBridge
from opendr.perception.object_detection_3d import VoxelObjectDetection3DLearner


class ObjectDetection3DVoxelNode:
    def __init__(
        self,
        input_point_cloud_topic="/opendr/dataset_point_cloud",
        detections_topic="/opendr/objects3d",
        device="cuda:0",
        model_name="tanet_car_xyres_16",
        model_config_path=os.path.join(
            "$OPENDR_HOME", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "tanet",
            "ped_cycle", "test_short.proto"
        ),
        temp_dir="temp",
    ):
        """
        Creates a ROS Node for 3D object detection
        :param input_point_cloud_topic: Topic from which we are reading the input point cloud
        :type input_point_cloud_topic: str
        :param detections_topic: Topic to which we are publishing the annotations
        :type detections_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model_name: the pretrained model to download or a trained model in temp_dir
        :type model_name: str
        :param temp_dir: where to store models
        :type temp_dir: str
        """

        self.learner = VoxelObjectDetection3DLearner(
            device=device, temp_path=temp_dir, model_config_path=model_config_path
        )
        if not os.path.exists(os.path.join(temp_dir, model_name)):
            VoxelObjectDetection3DLearner.download(model_name, temp_dir)

        self.learner.load(os.path.join(temp_dir, model_name), verbose=True)

        self.input_point_cloud_topic = input_point_cloud_topic
        self.bridge = ROSBridge()

        self.detection_publisher = rospy.Publisher(
            detections_topic, Detection3DArray, queue_size=1
        )

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        point_cloud = self.bridge.from_ros_point_cloud(data)
        detection_boxes = self.learner.infer(point_cloud)

        # Convert detected boxes to ROS type and publish
        ros_boxes = self.bridge.to_ros_boxes_3d(detection_boxes, classes=["Car", "Van", "Truck", "Pedestrian", "Cyclist"])
        self.detection_publisher.publish(ros_boxes)

    def listen(self):
        """
        Start the node and begin processing input data.
        """
        rospy.init_node('opendr_object_detection_3d_voxel_node', anonymous=True)
        rospy.Subscriber(self.input_point_cloud_topic, ROS_PointCloud, self.callback, queue_size=1, buff_size=10000000)

        rospy.loginfo("Object Detection 3D Voxel Node started.")
        rospy.spin()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_point_cloud_topic",
                        help="Point Cloud topic provided by either a point_cloud_dataset_node or any other 3D Point Cloud Node",
                        type=str, default="/opendr/dataset_point_cloud")
    parser.add_argument("-d", "--detections_topic",
                        help="Output detections topic",
                        type=str, default="/opendr/objects3d")
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("-n", "--model_name", help="Name of the trained model",
                        type=str, default="tanet_car_xyres_16", choices=["tanet_car_xyres_16"])
    parser.add_argument(
        "-c", "--model_config_path", help="Path to a model .proto config",
        type=str, default=os.path.join(
            "$OPENDR_HOME", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "tanet",
            "car", "xyres_16.proto"
        )
    )
    parser.add_argument("-t", "--temp_dir", help="Path to a temporary directory with models",
                        type=str, default="temp")
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

    voxel_node = ObjectDetection3DVoxelNode(
        device=device,
        model_name=args.model_name,
        model_config_path=args.model_config_path,
        input_point_cloud_topic=args.input_point_cloud_topic,
        temp_dir=args.temp_dir,
        detections_topic=args.detections_topic,
    )

    voxel_node.listen()


if __name__ == '__main__':
    main()
