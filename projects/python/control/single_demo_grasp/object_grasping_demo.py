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


import sys
import copy
import rospy
import numpy as np
import time
import tf
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import random
from math import pi
from std_msgs.msg import Int16, Float32MultiArray
from geometry_msgs.msg import PoseStamped



    
# executing a sequence of actions to demonstrate a grasping action based on
# single demo grasp model
def main():
    try:
        print("============ Initializing panda control ...")
        Commander = SingleDemoGraspAction()

        # reaching home position above objects with grippers opened
        print("============ Press `Enter` to reach home pose ...")
        input()
        Commander.home_pose()

        # send a request detection to detection server and waits until it receives a reply
        print("press enter to send a detection request to detector node")
        input()
        Commander.request_detection()

        # after receiving the detections, checks if there is an object found in the
        # image frame, and brings the robot's camera view above the object and correct
        # the object's orientation (in case some objects might be placed in the edges
        # and could be hardly visible) to have better predictions.
        if Commander.detections[2] != 1e10:
            Commander.reach_hover()
            Commander.fix_angle(Commander.detections[2])

        # another request, to receive the exact location of grasp followed by translating
        # the 2D coordinate (position in image frame) to 3D (corresponding position
        # in world frame) and executing the grasping action.
        print("============ Press `Enter` to find and reach grasp location")
        input()
        Commander.request_detection()
        if Commander.detections[0] != 1e10 or Commander.detections[1] != 1e10:
            Commander.reach_grasp_hover_kps()

        # lifting the object
        print("give input to close the gripper and lift the object")
        input()
        Commander.close_hand()
        cartesian_plan, fraction = Commander.plan_linear_z(0.45)
        Commander.execute_plan(cartesian_plan)
        print("============ Python Commander demo complete!")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':

    rospy.init_node('SingleDemoGraspAction', anonymous=True)
    listener_tf = tf.TransformListener()
    main()
    sys.exit()