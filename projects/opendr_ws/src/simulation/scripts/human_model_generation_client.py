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
import cv2
import os
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from simulation.srv import Mesh_vc

if __name__ == '__main__':
    rgb_img = cv2.imread(os.path.join(os.environ['OPENDR_HOME'], 'src/opendr/simulation/\
                                      human_model_generation/imgs_input/rgb/result_0004.jpg'))
    msk_img = cv2.imread(os.path.join(os.environ['OPENDR_HOME'], 'src/opendr/simulation/\
                                      human_model_generation/imgs_input/msk/result_0004.jpg'))
    bridge = CvBridge()
    rgb_img_msg = bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
    msk_img_msg = bridge.cv2_to_imgmsg(msk_img, encoding="bgr8")
    rospy.wait_for_service('human_model_generation')
    try:
        human_model_gen = rospy.ServiceProxy('human_model_generation', Mesh_vc)
        extract_pose = Bool()
        extract_pose.data = True
        mesh_vc = human_model_gen(rgb_img_msg, msk_img_msg, extract_pose)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
