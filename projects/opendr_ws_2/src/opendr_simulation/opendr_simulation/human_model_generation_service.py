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
import torch
import numpy as np
from opendr_ros2_bridge import ROS2Bridge
from opendr.simulation.human_model_generation.pifu_generator_learner import PIFuGeneratorLearner
from opendr_interfaces.srv import Mesh
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class Pifu_service(Node):

    def __init__(self, service_name='human_model_generation', device="cuda", checkpoint_dir='.'):
        """
        Creates a ROS Service for human model generation
        :param service_name: The name of the service
        :type service_name: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param checkpoint_dir: the directory where the PIFu weights will be downloaded/loaded
        :type checkpoint_dir: str
        """
        super().__init__('human_model_generation_service')
        self.bridge = ROS2Bridge()
        self.service_name = service_name
        # Initialize the pose estimation
        self.model_generator = PIFuGeneratorLearner(device=device, checkpoint_dir=checkpoint_dir)
        my_callback_group = MutuallyExclusiveCallbackGroup()

        self.srv = self.create_service(Mesh, 'human_model_generation', self.gen_callback, callback_group=my_callback_group)

    def gen_callback(self, request, response):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        rgb_img = self.bridge.from_ros_image(request.rgb_img)
        msk_img = self.bridge.from_ros_image(request.msk_img)
        extract_pose = request.extract_pose.data
        output = self.model_generator.infer([rgb_img], [msk_img], extract_pose=True)
        pose = np.zeros([18, 3])
        if extract_pose is True:
            model_3D = output[0]
            pose = output[1]
        else:
            model_3D = output
        verts = model_3D.get_vertices()
        faces = model_3D.get_faces()
        vert_colors = model_3D.vert_colors
        response.mesh = self.bridge.to_ros_mesh(verts, faces)
        response.vertex_colors = self.bridge.to_ros_colors(vert_colors)
        response.pose = self.bridge.to_ros_3Dpose(pose)
        return response


def main():
    # Select the device for running the
    try:
        if torch.cuda.is_available():
            print("GPU found.")
            device = 'cuda'
        else:
            print("GPU not found. Using CPU instead.")
            device = 'cpu'
    except:
        device = 'cpu'
    rclpy.init()
    pifu_service = PifuService(device=device)
    rclpy.spin(pifu_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
