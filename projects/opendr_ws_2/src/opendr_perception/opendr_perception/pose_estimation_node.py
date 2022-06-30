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

import cv2
import torch
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROS_Image
from vision_msgs.msg import Detection2DArray
from opendr_ros2_bridge import ROS2Bridge

from opendr.engine.data import Image
from opendr.perception.pose_estimation import draw
from opendr.perception.pose_estimation import LightweightOpenPoseLearner


class PoseEstimationNode(Node):

    def __init__(self, input_image_topic="image_raw", output_image_topic="/opendr/image_pose_annotated",
                 pose_annotations_topic="/opendr/poses", device="cuda"):
        super().__init__('pose_estimation_node')

        if output_image_topic is not None:
            self.image_publisher = self.create_publisher(ROS_Image, output_image_topic, 1)
        else:
            self.image_publisher = None

        if pose_annotations_topic is not None:
            self.pose_publisher = self.create_publisher(Detection2DArray, pose_annotations_topic, 1)
        else:
            self.pose_publisher = None

        self.image_subscriber = self.create_subscription(ROS_Image, input_image_topic, self.callback, 1)

        self.bridge = ROS2Bridge()

        self.pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=0,
                                                         mobilenet_use_stride=False,
                                                         half_precision=False)
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")

    def callback(self, data):
        image = self.bridge.from_ros_image(data, encoding='bgr8')
        cv2.imshow("image", image.opencv())
        cv2.waitKey(5)

        poses = self.pose_estimator.infer(image)

        image = image.opencv()
        #  Annotate image and publish results
        for pose in poses:
            if self.pose_publisher is not None:
                self.pose_publisher.publish(self.bridge.to_ros_pose(pose))
            draw(image, pose)

        if self.image_publisher is not None:
            # Convert OpenDR image to ROS image message to publish
            message = self.bridge.to_ros_image(Image(image), encoding='bgr8')
            self.image_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    try:
        if torch.cuda.is_available():
            print("GPU found.")
            device = 'cuda'
        else:
            print("GPU not found. Using CPU instead.")
            device = 'cpu'
    except:
        device = 'cpu'

    pose_estimator_node = PoseEstimationNode(device=device)

    rclpy.spin(pose_estimator_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_estimator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
