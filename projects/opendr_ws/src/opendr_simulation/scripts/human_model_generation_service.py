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


import rospy
import torch
import numpy as np
from opendr_bridge import ROSBridge
from opendr.simulation.human_model_generation.pifu_generator_learner import PIFuGeneratorLearner
from opendr_simulation.srv import Mesh_vc


class PifuNode:

    def __init__(self, service_name='human_model_generation', device="cuda", checkpoint_dir='.'):
        """
        Creates a ROS Node for pose detection
        :param 'human_model_generation': The name of the service
        :type input_image_topic: str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param checkpoint_dir: the directory where the PIFu weights will be downloaded/loaded
        :type checkpoint_dir: str
        """

        self.bridge = ROSBridge()
        self.service_name = service_name
        # Initialize the pose estimation
        self.model_generator = PIFuGeneratorLearner(device=device, checkpoint_dir=checkpoint_dir)

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_human_model_generation', anonymous=True)
        rospy.Service('human_model_generation', Mesh_vc, self.handle_human_model_generation)
        rospy.loginfo("Human model generation node started!")
        rospy.spin()

    def handle_human_model_generation(self, msg):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        rgb_img = self.bridge.from_ros_image(msg.rgb_img)
        msk_img = self.bridge.from_ros_image(msg.msk_img)
        extract_pose = msg.extract_pose.data
        output = self.model_generator.infer([rgb_img], [msk_img], extract_pose=extract_pose)
        pose = np.zeros([18, 3])
        if extract_pose is True:
            model_3D = output[0]
            pose = output[1]
        else:
            model_3D = output
        verts = model_3D.get_vertices()
        faces = model_3D.get_faces()
        vert_colors = model_3D.vert_colors
        msg_mesh = self.bridge.to_ros_mesh(verts, faces)
        msg_v_colors = self.bridge.to_ros_colors(vert_colors)
        msg_pose = self.bridge.to_ros_3Dpose(pose)
        return msg_mesh, msg_v_colors, msg_pose

if __name__ == '__main__':
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

    pifuNode = PifuNode(device=device)
    pifuNode.listen()
