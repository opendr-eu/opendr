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
import cv2
import os
from cv_bridge import CvBridge
from opendr_bridge import ROSBridge
from std_msgs.msg import Bool
from opendr_simulation.srv import Mesh_vc
from opendr.simulation.human_model_generation.utilities.model_3D import Model_3D


if __name__ == '__main__':
    rgb_img = cv2.imread(os.path.join(os.environ['OPENDR_HOME'], 'projects/python/simulation/'
                                      'human_model_generation/demos/imgs_input/rgb/result_0004.jpg'))
    msk_img = cv2.imread(os.path.join(os.environ['OPENDR_HOME'], 'projects/python/simulation/'
                                      'human_model_generation/demos/imgs_input/msk/result_0004.jpg'))
    bridge_cv = CvBridge()
    bridge_ros = ROSBridge()
    rgb_img_msg = bridge_cv.cv2_to_imgmsg(rgb_img, encoding="bgr8")
    msk_img_msg = bridge_cv.cv2_to_imgmsg(msk_img, encoding="bgr8")
    srv_name = 'human_model_generation'
    rospy.wait_for_service(srv_name)
    try:
        human_model_gen = rospy.ServiceProxy(srv_name, Mesh_vc)
        extract_pose = Bool()
        extract_pose.data = True
        client_msg = human_model_gen(rgb_img_msg, msk_img_msg, extract_pose)
        pose = bridge_ros.from_ros_3Dpose(client_msg.pose)
        vertices, triangles = bridge_ros.from_ros_mesh(client_msg.mesh)
        vertex_colors = bridge_ros.from_ros_colors(client_msg.vertex_colors)
        human_model = Model_3D(vertices, triangles, vertex_colors)
        human_model.save_obj_mesh('./human_model.obj')
        [out_imgs, human_pose_2D] = human_model.get_img_views(rotations=[30, 120], human_pose_3D=pose, plot_kps=True)
        cv2.imwrite('./rendering.png', out_imgs[0].opencv())
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
