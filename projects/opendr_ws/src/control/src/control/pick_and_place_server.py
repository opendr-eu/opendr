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

import rospy
import actionlib

from control.msg import PickResult, PlaceResult, PickAction, PlaceAction


class PickAndPlaceServer(object):
    def __init__(self, rotate_EE, stop_action, resume_action,
                    move_joint_space, move_cartesian_space,
                    move_cartesian_space_1D, move_cartesian_space_2D,
                    grasp, move_gripper):
        self.rotate_EE = rotate_EE
        self.stop_action = stop_action
        self.resume_action = resume_action
        self.move_joint_space = move_joint_space
        self.move_cartesian_space = move_cartesian_space
        self.move_cartesian_space_1D = move_cartesian_space_1D
        self.move_cartesian_space_2D = move_cartesian_space_2D
        self.grasp = grasp
        self.move_gripper = move_gripper

        self.pick_server = actionlib.SimpleActionServer('/opendr/pick',
                                                            PickAction,
                                                            self.pick,
                                                            auto_start=False)

        self.place_server = actionlib.SimpleActionServer('/opendr/place',
                                                            PlaceAction,
                                                            self.place,
                                                            auto_start=False)

        self.pause_sub = rospy.Subscriber('request_pause', Empty, self.request_pause)
        self.resume_sub = rospy.Subscriber('request_resume', Empty, self.request_resume)

    def __del__(self):
        self.pick_server = None
        self.place_server = None

    def start(self):
        self._loginfo('PickAndPlace Server Started')
        self.pick_server.start()
        self.place_server.start()

    def stop(self):
        self._loginfo('PickAndPlace Server Stopped')
        self.pick_server = None
        self.place_server = None

    def pick(self, goal):
        # type: (DoSomethingGoal) -> None
        self._loginfo('PickAndPlace Server received do_something pick() request')
        success = True
        z_final = goal.pose.position.z
        z_intermediate = goal.pose.position.z + 0.3
        goal.pose.position.z = z_intermediate
        # Aproach
        self.move_cartesian_space(goal.pose)
        # Pre-grasp
        self.move_cartesian_space_1D(z_final)
        # Grasp
        self.grasp(Grasp(width= , force= , pose=))
        # Post-grasp
        self.move_cartesian_space_1D(z_intermediate)

        if self.pick_server.is_preempt_requested():
            self.pick_server.set_preempted()
            success = False
            break

        if success:
            self._loginfo('do_something action succeeded')
            result = PickResult()
            self.pick_server.set_succeeded(result)

    def place(self, goal):
        # type: (DoSomethingGoal) -> Non
        self._loginfo('PickAndPlace server received place() request')
        success = True


        if self.place_server.is_preempt_requested():
            self._loginfo('do_something_else action preempted')
            self.place_server.set_preempted()
            success = False
            break

        if success:
            self._loginfo('do_something_else action succeeded')
            result = PlaceResult()
            self.place_server.set_succeeded(result)

    def request_pause(self, msg):
        self.stop_action()

    def request_resume(self, msg):
        self.resume_action()

    @staticmethod
    def _loginfo(message):
        # type: (str) -> None
        rospy.loginfo('PickAndPlaceServer ({}) {}'.format('opendr_example', message))
