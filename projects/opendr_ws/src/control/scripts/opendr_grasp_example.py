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
from control.detections import Detections
from control.pick_and_place_client import PickAndPlaceClient
from control.msgs import Grasp
from geometry_msgs.msgs import Pose


if __name__ == '__main__':
    rospy.init_node('opendr_grasp_example', anonymous=True)

    detections = Detections()
    detection_sub = rospy.Subscriber('/opendr/grasp_detected', ObjectHypothesisWithPose, detections.process_detection)
    obj_cat_sub = rospy.Subscriber("/opendr/object_catagories", VisionInfo, detections.save_categories)

    pick_and_place_client = PickAndPlaceClient()
    pick_and_place_client.start()

    def stop_pick_and_place_client():
        pick_and_place_client.stop()

    rospy.on_shutdown(stop_pick_and_place_client)

    rospy.loginfo("opendr_grasp_example node started!")

    # Create grasp goal
    grasp_msg = Grasp()
    pushrod_id = detections.find_object_by_category("pushrod")
    pushrod_pose = detections.get_object_pose(pushrod_id)
    grasp_msg.pose = pushrod_pose
    grasp_msg.width = 0.01
    grasp_msg.force = 20.0
    # Send pick goal to server
    pick_and_place_client.pick(grasp_msg)
    # Create place goal
    place_msg = Pose()
    pushrod_hole_id = detections.find_object_by_category("small pushrod holes")
    pushrod_hole_pose = detections.get_object_pose(pushrod_hole_id)
    # Send place goal to server
    pick_and_place_client.place(pushrod_hole_pose)

    rospy.spin()
