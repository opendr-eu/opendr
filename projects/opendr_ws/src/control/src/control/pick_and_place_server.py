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


import time
import tf
import rospy
import actionlib
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
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
        self._counter = 0
        self.pick_server = actionlib.SimpleActionServer('/opendr/pick',
                                                        PickAction,
                                                        self.pick2,
                                                        auto_start=False)

        self.place_server = actionlib.SimpleActionServer('/opendr/place',
                                                         PlaceAction,
                                                         self.place,
                                                         auto_start=False)

        self.pause_sub = rospy.Subscriber('/opendr/commands', Bool, self.request_pause)
        self._pause = False
        self._table_level = 0.115

    def __del__(self):
        self.pick_server = None
        self.place_server = None

    def start(self):
        self._loginfo('PickAndPlace Server Started')
        self.pick_server.start()
        self.place_server.start()
        self.move_gripper(0.08)
        self.move_gripper(0.04)

    def stop(self):
        self._loginfo('PickAndPlace Server Stopped')
        self.pick_server = None
        self.place_server = None

    def pick(self, goal):
        self._loginfo('PickAndPlace Server received do_something pick() request')
        success = True
        z_final = goal.pose.position.z if goal.pose.position.z > self._table_level else self._table_level
        z_intermediate = z_final + 0.15
        print("zs")
        print(z_final)
        print(z_intermediate)
        # Aproach
        self.move_cartesian_space_2D([goal.pose.position.x, goal.pose.position.y], False)
        self.move_cartesian_space_1D(z_intermediate, False)
        orientation_list = [goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        self.rotate_EE(yaw)
        # Pre-grasp
        self.move_cartesian_space_1D(z_final, True)
        # Grasp
        self.grasp(goal.width, goal.force)
        # Post-grasp
        self.move_cartesian_space_1D(z_intermediate, False)

        if success:
            self._loginfo('do_something action succeeded')
            result = PickResult(True)
            self.pick_server.set_succeeded(result)

    def pick2(self, goal):
        self._loginfo('PickAndPlace Server received do_something pick() request')
        success = True
        z_final = goal.pose.position.z if goal.pose.position.z > self._table_level else self._table_level
        z_intermediate = z_final + 0.15
        print("zs")
        print(z_final)
        print(z_intermediate)
        pose_intermediate = goal.pose
        pose_intermediate.position.z = z_intermediate
        # Aproach
        self.move_cartesian_space(pose_intermediate)
        # Pre-grasp
        self.move_cartesian_space_1D(z_final, True)
        # Grasp
        self.grasp(goal.width, goal.force)
        # Post-grasp
        self.move_cartesian_space_1D(z_intermediate, False)

        if success:
            self._loginfo('do_something action succeeded')
            result = PickResult(True)
            self.pick_server.set_succeeded(result)

    def place(self, goal):
        self._loginfo('PickAndPlace server received place() request')
        success = True

        z_final = goal.pose.position.z
        z_intermediate = goal.pose.position.z + 0.3
        goal.pose.position.z = z_intermediate

        if self._counter < 3:
            prepose = Pose()
            prepose.position.x = 0.4955222156145186
            prepose.position.y = -0.11675839207592095
            prepose.position.z = 0.3986073776445874
            prepose.orientation.x = -0.6609967852858333
            prepose.orientation.y = -0.31114417228114366
            prepose.orientation.z = -0.27521515925565554
            prepose.orientation.w = 0.6249233313080573
            self.move_cartesian_space(prepose)

        if self._counter < 6:
            self.move_cartesian_space(goal.pose)

            self.move_cartesian_space_1D(z_final, True)

            if not self._counter < 3:
                time.sleep(2)
            self.move_gripper(0.04)

            self.move_cartesian_space_1D(z_intermediate, False)
        else:
            goal.pose.position.z = z_final
            self.move_cartesian_space(goal.pose)
            time.sleep(1)
            self.move_gripper(0.04)
        if success:
            self._counter += 1
            self._loginfo('do_something_else action succeeded')
            result = PlaceResult(True)
            self.place_server.set_succeeded(result)

    def request_pause(self, msg):
        if msg.data and not self._pause:
            self.stop_action()
            self._pause = True
        elif not msg.data and self._pause:
            self.resume_action()
            self._pause = False

    @staticmethod
    def _loginfo(message):
        rospy.loginfo('PickAndPlaceServer ({}) {}'.format('opendr_example', message))
