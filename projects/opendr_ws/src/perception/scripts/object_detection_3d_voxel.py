#!/usr/bin/env python
# Copyright 2020-2021 OpenDR European Project
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
import os
import rospy
from vision_msgs.msg import Detection3DArray
from sensor_msgs.msg import PointCloud as ROS_PointCloud
from opendr_bridge import ROSBridge
from opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
)


class ObjectDetection3DVoxelNode:
    def __init__(
        self,
        input_point_cloud_topic="/opendr/dataset_point_cloud",
        output_detection3d_topic="/opendr/detection3d",
        device="cuda:0",
        model_name="tanet_car_xyres_16",
        model_config_path=os.path.join(
            "..", "..", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "tanet",
            "ped_cycle", "test_short.proto"
        ),
        temp_dir="temp",
    ):
        """
        Creates a ROS Node for 3D object detection
        :param input_point_cloud_topic: Topic from which we are reading the input point cloud
        :type input_image_topic: str
        :param output_detection3d_topic: Topic to which we are publishing the annotations
        :type output_detection3d_topic:  str
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

        # Initialize OpenDR ROSBridge object
        self.bridge = ROSBridge()

        self.detection_publisher = rospy.Publisher(
            output_detection3d_topic, Detection3DArray, queue_size=10
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
        detection_boxes = self.learner.infer(point_cloud)

        # Convert detected boxes to ROS type and publish
        ros_boxes = self.bridge.to_ros_boxes_3d(detection_boxes, classes=["Car", "Van", "Truck", "Pedestrian", "Cyclist"])
        if self.detection_publisher is not None:
            self.detection_publisher.publish(ros_boxes)
            rospy.loginfo("Published detection boxes")

if __name__ == "__main__":
    # Automatically run on GPU/CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # initialize ROS node
    rospy.init_node("opendr_voxel_detection_3d", anonymous=True)
    rospy.loginfo("Voxel Detection 3D node started")

    model_name = rospy.get_param("~model_name", "tanet_car_xyres_16")
    model_config_path = rospy.get_param(
        "~model_config_path", os.path.join(
            "..", "..", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "tanet",
            "car", "test_short.proto"
        )
    )
    temp_dir = rospy.get_param("~temp_dir", "temp")
    input_point_cloud_topic = rospy.get_param(
        "~input_point_cloud_topic", "/opendr/dataset_point_cloud"
    )
    rospy.loginfo("Using model_name: {}".format(model_name))

    # created node object
    voxel_node = ObjectDetection3DVoxelNode(
        device=device,
        model_name=model_name,
        model_config_path=model_config_path,
        input_point_cloud_topic=input_point_cloud_topic,
        temp_dir=temp_dir,
    )
    # begin ROS communications
    rospy.spin()
