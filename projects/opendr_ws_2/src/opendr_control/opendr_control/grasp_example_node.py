#!/usr/bin/env python3
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

import time
import rospy
from control.detections import Detections
from control.pick_and_place_client import PickAndPlaceClient
from geometry_msgs.msg import Pose
from control.msg import PickGoal
from vision_msgs.msg import ObjectHypothesisWithPose, VisionInfo
import rospy
import actionlib
from control.msg import PickAction, PlaceAction, PlaceGoal, PickActionResult
from std_srvs.srv import Trigger
from control.srv import RotateEE, SetJointState, SetPoseTarget, SetPoseTarget2D, SetPoseTarget1D
from control.pick_and_place_server import PickAndPlaceServer


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


if __name__ == '__main__':
    rospy.init_node('opendr_grasp_example', anonymous=True)

    detections = Detections()
    detection_sub = rospy.Subscriber('/opendr/grasp_detected', ObjectHypothesisWithPose, detections.process_detection)
    obj_cat_sub = rospy.Subscriber("/opendr/object_categories", VisionInfo, detections.save_categories)

    pick_and_place_client = PickAndPlaceClient()
    pick_and_place_client.start()

    def stop_pick_and_place_client():
        pick_and_place_client.stop()

    rospy.on_shutdown(stop_pick_and_place_client)

    rospy.loginfo("opendr_grasp_example node started!")

    time.sleep(5)
    # Create grasp goal
    grasp_msg = PickGoal()
    pushrod_id = detections.find_object_by_category("pushrod")
    pushrod_pose = detections.get_object_pose(pushrod_id)
    while not pushrod_pose:
        pushrod_pose = detections.get_object_pose(pushrod_id)
    grasp_msg.pose = pushrod_pose
    grasp_msg.width = 0.008
    grasp_msg.force = 20.0
    # Send pick goal to server
    # pick_and_place_client.pick(grasp_msg)

    # Create place goal
    place_msg = Pose()
    pushrod_hole_id = detections.find_object_by_category("bolt holes")
    pushrod_hole_pose = detections.get_object_pose(pushrod_hole_id)
    # Send place goal to server
    # pick_and_place_client.place(pushrod_hole_pose)

    # Send pick and place goals to server
    pick_and_place_client.pick_and_place(grasp_msg, pushrod_hole_pose)
    rospy.spin()
