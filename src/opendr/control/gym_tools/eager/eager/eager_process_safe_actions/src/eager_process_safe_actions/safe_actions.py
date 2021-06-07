# Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import rospy
import moveit_commander
import gym
from eager_core.action_processor import ActionProcessor
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import CubicSpline
import numpy as np


class SafeActions(ActionProcessor):
    '''
    Custom action processing node for manipulators that are not allowed to move below a certain height.
    Can be used for collision avoidance, for example for manipulators that are mounted on a table or other sort of base surface.
    Checks velocity limits, prevents self-collision and collision with the base surface.
    
    !!!
    Be careful! Implementation assumes constant velocity based on the difference between the start and goal position and the duration parameter.
    If this assumption does not hold, velocity limit checking and collision avoidance will not work!
    !!!
    
    '''
    def __init__(self):
        # Get params
        object_frame = rospy.get_param('~object_frame')
        self.joint_names = rospy.get_param('~joint_names')
        self.group_name = rospy.get_param('~group_name')
        self.checks_per_rad = rospy.get_param('~checks_per_rad')
        self.vel_limit = rospy.get_param('~vel_limit')
        self.duration = rospy.get_param('~duration')
    	
        # Initialize Moveit Commander and Scene
        moveit_commander.roscpp_initialize(sys.argv)
        scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        
        # Add a collision object to the scenes
        p = PoseStamped()
        p.header.frame_id = object_frame
        p.pose.position.z = -0.05
        p.pose.orientation.w = 1
        scene.add_cylinder('base', p, 0.1, 1.5)
        
        # Initialize state validity check service
        self.state_validity_service = rospy.ServiceProxy('check_state_validity', GetStateValidity)
        self.state_validity_service.wait_for_service()
        
        super(SafeActions, self).__init__()
    
    def _get_space(self):
        space = gym.spaces.Box(low=-3.14, high=3.14, shape=(len(self.joint_names),))
        return space
        
    def _close(self):
        pass
    
    def _reset(self):
        pass

    def _process_action(self, action, observation):
        if len(observation) > 1:
            rospy.logwarn("[{}] Expected observation from only one robot".format(rospy.get_name()))
        for robot in observation:
            if len(observation[robot]) > 1:
                rospy.logwarn("[{}] Expected observation from only one sensor".format(rospy.get_name()))
            for sensor in observation[robot]:
                current_position = observation[robot][sensor]
        safe_action = self._getSafeAction(np.asarray(action), np.asarray(current_position))
        return safe_action

    def _getSafeAction(self, goal_position, current_position):
        '''
        Given a goal_position, check if this satisfies the velocity limit 
        and whether the path is collision free
        return a collision free action
        '''
        rs = RobotState()
        rs.joint_state.name = self.joint_names
        rs.joint_state.position = current_position

        gsvr = GetStateValidityRequest()
        gsvr.group_name = self.group_name
        gsvr.robot_state = rs

        # We interpolate using a cubic spline between the current joint state and the goal state
        # For now, we are not using velocity information, so it is actually linear interpolation
        # We could use the current velocity as a boundary condition for the spline
        t = [0, self.duration]
        x = [current_position, goal_position]
        cs = CubicSpline(t, x)

        # We also check joint limits on velocity
        dif = goal_position - current_position
        too_fast = np.abs(dif/self.duration) > self.vel_limit
        if np.any(too_fast):
            goal_position[too_fast] = current_position[too_fast] + \
                np.sign(dif[too_fast]) * self.duration * self.vel_limit
            x = [current_position, goal_position]
            cs = CubicSpline(t, x)

        max_angle_dif = np.max(np.abs(goal_position - current_position))

        n_checks = int(np.ceil(self.checks_per_rad * max_angle_dif))
        way_points = cs(np.linspace(0, self.duration, n_checks))
        
        for i in range(n_checks):
            gsvr.robot_state.joint_state.position = way_points[i, :]
            if not self.state_validity_service.call(gsvr).valid:
                return way_points[max(i-1, 0)]
        return goal_position
