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


import rospy
import torch
import numpy as np
from opendr_bridge import ROSBridge
from opendr.simulation.human_model_generation.pifu_generator_learner import PIFuGeneratorLearner
from simulation.srv import Mesh_vc
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


class PifuNode:

    def __init__(self, input_image_topic="/usb_cam/image_raw", output_human_mopdel_topic="/opendr/simulation/\
                 human_model_generation/human_model",
                 output_3Dpose__topic="/opendr/simulation/human_model_generation/", device="cuda"):
        """
        Creates a ROS Node for pose detection
        :param input_image_topic: Topic from which we are reading the input image
        :type input_image_topic: str
        :param output_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_image_topic: str
        :param pose_annotations_topic: Topic to which we are publishing the annotations (if None, we are not publishing
        annotated pose annotations)
        :type pose_annotations_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        """
        self.bridge = ROSBridge()
        # Initialize the pose estimation
        self.model_generator = PIFuGeneratorLearner(device='cuda', checkpoint_dir=".")

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
        msg_mesh, msg_v_colors = self.bridge.to_ros_mesh(verts, faces, vert_colors)
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
