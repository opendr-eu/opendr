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
    def __init__(self, robot_arm_namespace, gripper_namespace):
        self.pick_server = actionlib.SimpleActionServer('/opendr/pick',
                                                                PickAction,
                                                                self.pick,
                                                                auto_start=False)

        self.place_server = actionlib.SimpleActionServer('/opendr/place',
                                                                     PlaceAction,
                                                                     self.place,
                                                                     auto_start=False)


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

        r = rospy.Rate(1)

        start_time = time.time()

        while not rospy.is_shutdown():
            if time.time() - start_time > goal.how_long_to_do_something:
                break

            if self.pick_server.is_preempt_requested():
                self._loginfo('do_something action preempted')

                self.pick_server.set_preempted()
                success = False
                break

            self._loginfo('Doing something action')
            r.sleep()

        if success:
            self._loginfo('do_something action succeeded')
            result = PickResult()
            result.did_finish_doing_something = True
            self.pick_server.set_succeeded(result)

    def place(self, goal):
        # type: (DoSomethingGoal) -> None
        
        self._loginfo('PickAndPlace server received place() request')
        success = True

        r = rospy.Rate(1)

        start_time = time.time()

        while not rospy.is_shutdown():
            if time.time() - start_time > goal.how_long_to_do_something:
                break

            if self.place_server.is_preempt_requested():
                self._loginfo('do_something_else action preempted')

                self.place_server.set_preempted()
                success = False
                break

            self._loginfo('Doing something_else action')
            r.sleep()

        if success:
            self._loginfo('do_something_else action succeeded')
            result = PlaceResult()
            result.did_finish_doing_something = True
            self.place_server.set_succeeded(result)

    @staticmethod
    def _loginfo(message):
        # type: (str) -> None

        rospy.loginfo('PickAndPlaceServer ({}) {}'.format('dummy_server', message))


