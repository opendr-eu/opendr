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

import rclpy
from rclpy.node import Node

import cv2
import os
from cv_bridge import CvBridge
from opendr_ros2_bridge import ROS2Bridge
from std_msgs.msg import Bool
from opendr_interfaces.srv import Mesh
from opendr.simulation.human_model_generation.utilities.model_3D import Model_3D


class HumanModelGenerationClient(Node):

    def __init__(self, service_name='human_model_generation'):
        super().__init__('human_model_generation_client')
        self.bridge_cv = CvBridge()
        self.bridge_ros = ROS2Bridge()
        self.cli = self.create_client(Mesh, service_name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Mesh.Request()

    def send_request(self, rgb_img, msk_img, extract_pose):
        extract_pose_ros = Bool()
        extract_pose_ros.data = extract_pose
        self.req.rgb_img = self.bridge_cv.cv2_to_imgmsg(rgb_img, encoding="bgr8")
        self.req.msk_img = self.bridge_cv.cv2_to_imgmsg(msk_img, encoding="bgr8")
        self.req.extract_pose = extract_pose_ros
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        resp = self.future.result()
        pose = self.bridge_ros.from_ros_3Dpose(resp.pose)
        vertices, triangles = self.bridge_ros.from_ros_mesh(resp.mesh)
        vertex_colors = self.bridge_ros.from_ros_colors(resp.vertex_colors)
        human_model = Model_3D(vertices, triangles, vertex_colors)
        return human_model, pose


def main():
    rgb_img = cv2.imread(os.path.join(os.environ['OPENDR_HOME'], 'projects/simulation/'
                                      'human_model_generation/demos/imgs_input/rgb/result_0004.jpg'))
    msk_img = cv2.imread(os.path.join(os.environ['OPENDR_HOME'], 'projects/simulation/'
                                      'human_model_generation/demos/imgs_input/msk/result_0004.jpg'))
    extract_pose = True
    rclpy.init()
    client = Human_model_generation_client()
    [human_model, pose] = client.send_request(rgb_img, msk_img, extract_pose=extract_pose)
    human_model.save_obj_mesh('./human_model.obj')
    if extract_pose:
        [out_imgs, _] = human_model.get_img_views(rotations=[30, 120], human_pose_3D=pose, plot_kps=True)
    else:
        [out_imgs, _] = human_model.get_img_views(rotations=[30, 120], human_pose_3D=None, plot_kps=False)
    cv2.imwrite('./rendering.png', out_imgs[0].opencv())
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
