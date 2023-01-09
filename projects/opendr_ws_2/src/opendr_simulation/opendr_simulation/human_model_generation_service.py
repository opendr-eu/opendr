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

import rclpy
from rclpy.node import Node
import argparse
import os
import torch
from opendr_bridge import ROS2Bridge
from opendr.simulation.human_model_generation.pifu_generator_learner import PIFuGeneratorLearner
from opendr_interface.srv import ImgToMesh
from opendr.engine.target import Pose
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class PifuService(Node):

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

        self.srv = self.create_service(ImgToMesh, 'human_model_generation', self.gen_callback, callback_group=my_callback_group)

    def gen_callback(self, request, response):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param request: The service request
        :type request: SrvTypeRequest
        :param response: SrvTypeResponse
        :type response: The service response
        :return response: SrvTypeResponse
        :type response: The service response
        """
        img_rgb = self.bridge.from_ros_image(request.img_rgb)
        img_msk = self.bridge.from_ros_image(request.img_msk)
        extract_pose = request.extract_pose.data
        output = self.model_generator.infer([img_rgb], [img_msk], extract_pose=extract_pose)
        if extract_pose is True:
            model_3D = output[0]
            pose = output[1]
        else:
            model_3D = output
            pose = Pose([], 0.0)
        verts = model_3D.get_vertices()
        faces = model_3D.get_faces()
        vert_colors = model_3D.vert_colors
        response.mesh = self.bridge.to_ros_mesh(verts, faces)
        response.vertex_colors = self.bridge.to_ros_colors(vert_colors)
        response.pose = self.bridge.to_ros_pose_3D(pose)
        return response


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use, either \"cpu\" or \"cuda\", defaults to \"cuda\"",
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--srv_name", help="The name of the service",
                        type=str, default="human_model_generation")
    parser.add_argument("--checkpoint_dir", help="Path to directory for the checkpoints of the method's network",
                        type=str, default=os.path.join(os.environ['OPENDR_HOME'], 'projects/opendr_ws_2'))
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

    rclpy.init()
    pifu_service = PifuService(service_name=args.srv_name, device=device, checkpoint_dir=args.checkpoint_dir)
    rclpy.spin(pifu_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
