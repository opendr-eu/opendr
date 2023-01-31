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
import actionlib
from control.msg import PickAction, PlaceAction, PlaceGoal, PickActionResult


class PickAndPlaceClient(object):
    def __init__(self):
        self.pick_client = actionlib.SimpleActionClient('/opendr/pick',
                                                        PickAction)
        self.place_client = actionlib.SimpleActionClient('/opendr/place',
                                                         PlaceAction)

    def start(self):
        self.pick_client.wait_for_server(rospy.Duration.from_sec(15))
        self.place_client.wait_for_server(rospy.Duration.from_sec(15))
        self._loginfo('Action Client Started')

    def stop(self):
        self._loginfo('Action Client Stopped')
        self.pick_client = None
        self.place_client = None

    def pick(self, grasp_msg):
        try:
            self.pick_client.send_goal(grasp_msg,
                                       active_cb=self._pick_active,
                                       feedback_cb=self._pick_feedback,
                                       done_cb=self._pick_done)

            self._loginfo('pick has been sent')
        except Exception as e:
            print(e)
            self._loginfo('Error sending pick')

    def _pick_active(self):
        self._loginfo('Pick has transitioned to active state')

    def _pick_feedback(self, feedback):
        self._loginfo('Pick feedback received: {}'.format(feedback))

    def _pick_done(self, state, result):
        self._loginfo('Pick done callback triggered')
        self._loginfo(str(state))
        self._loginfo(str(result))

    def place(self, pose_msg):
        try:
            goal = PlaceGoal(pose=pose_msg)
            self.place_client.send_goal(goal,
                                        active_cb=self.place_active,
                                        feedback_cb=self.place_feedback,
                                        done_cb=self.place_done)
            self._loginfo('place has been sent')
        except Exception as e:
            print(e)
            self._loginfo('Error sending place')

    def place_active(self):
        self._loginfo('Place has transitioned to active state')

    def place_feedback(self, feedback):
        self._loginfo('Place feedback received: {}'.format(feedback))

    def place_done(self, state, result):
        self._loginfo('Place done callback triggered')
        self._loginfo(str(state))
        self._loginfo(str(result))

    def pick_and_place(self, pick_goal, place_goal):
        self.pick(pick_goal)
        pick_result = rospy.wait_for_message("/opendr/pick/result",
                                             PickActionResult)
        if pick_result.result.success:
            self.place(place_goal)
            rospy.wait_for_message("/opendr/place/result", PickActionResult)

    @staticmethod
    def _loginfo(message):
        # type: (str) -> None
        rospy.loginfo('PickAndPlaceClient ({}) {}'.format('opendr_example',
                                                          message))
