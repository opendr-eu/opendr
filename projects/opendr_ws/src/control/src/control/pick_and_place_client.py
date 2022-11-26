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

import rospy
import actionlib
from std_msgs.msg import Empty
from control.msg import PickAction, PlaceAction, PickGoal, PlaceGoal


class PickAndPlaceClient(object):
    def __init__(self):

        self.pick_client = actionlib.SimpleActionClient('/opendr/pick', PickAction)

        self.place_client = actionlib.SimpleActionClient('/opendr/place', PlaceAction)

    def start(self):
        self._loginfo('Action Client Started')
        self.pick_client.wait_for_server(rospy.Duration.from_sec(15))
        self.place_client.wait_for_server(rospy.Duration.from_sec(15))

    def stop(self):
        self._loginfo('Action Client Stopped')
        self.pick_client = None
        self.place_client = None

    def pick(self, length):
        try:
            goal = DoSomethingGoal()
            goal.how_long_to_do_something = length
            self.pick_client.send_goal(goal,
                                        active_cb=self._pick_active,
                                        feedback_cb=self.pick_feedback,
                                        done_cb=self.pick_done)

            self._loginfo('pick has been sent')

        except Exception as e:
            print(e.message)
            traceback.print_exc()
            self._loginfo('Error sending pick')
            self.pick_client.wait_for_server(rospy.Duration.from_sec(15))
            self.sendpick(length)

    def pick_active(self):
        self._loginfo('Pick has transitioned to active state')

    def pick_feedback(self, feedback):
        # type: (DoSomethingFeedback) -> None
        self._loginfo('Pick feedback received: {}'.format(feedback))

    def pick_done(self, state, result):
        # type: (actionlib.GoalStatus, DoSomethingResult) -> None
        self._loginfo('Pick done callback triggered')
        self._loginfo(str(state))
        self._loginfo(str(result))
        self._loginfo('Do something result: ' + str(result.did_finish_doing_something))

    def place(self, length):
        try:
            goal = DoSomethingGoal()
            goal.how_long_to_do_something = length
            self.place_client.send_goal(goal,
                                                    active_cb=self.place_active,
                                                    feedback_cb=self.place_feedback,
                                                    done_cb=self.place_done)

        except Exception as e:
            print(e.message)
            traceback.print_exc()
            self._loginfo('Error sending place')
            self.place_client.wait_for_server(rospy.Duration.from_sec(15))
            self.pick(length)

    def place_active(self):
        self._loginfo('Place has transitioned to active state')

    def place_feedback(self, feedback):
        # type: (DoSomethingFeedback) -> None
        self._loginfo('Place feedback received: {}'.format(feedback))

    def place_done(self, state, result):
        # type: (actionlib.GoalStatus, DoSomethingResult) -> None
        self._loginfo('Place done callback triggered')
        self._loginfo(str(state))
        self._loginfo(str(result))
        self._loginfo('Do Place result: ' + str(result.did_finish_doing_something))

    def _trigger_one(self, data):
        self._loginfo('Start pick action')
        self._loginfo(data)

        self.pick(30)

    def _trigger_two(self, data):
        self._loginfo('Start place action')
        self._loginfo(data)

        self.place(15)

    @staticmethod
    def _loginfo(message):
        # type: (str) -> None

        rospy.loginfo('DummyActionClient ({}) {}'.format('dummy_client', message))
