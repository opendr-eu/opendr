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


import rospy
import numpy as np
from opendr.planning.end_to_end_planning import EndToEndPlanningRLLearner, AgiEnv
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
import os


class EndToEndPlannerNode:

    def __init__(self):
        """
        Creates a ROS Node for pose detection
        """
        self.node_name = "opendr_end_to_end_planner"
        self.input_depth_image_topic = "/range_image_raw"
        self.pose_topic = "/mavros/local_position/pose"
        self.current_position = PoseStamped().pose.position
        self.env = AgiEnv()
        self.end_to_end_planner = EndToEndPlanningRLLearner(self.env)
        print(os.environ.get("OPENDR_HOME"))
        self.end_to_end_planner.load(os.environ.get("OPENDR_HOME") + '/src/opendr/planning/end_to_end_planning/pretrained_model/saved_model.zip')

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_end_to_end_planner', anonymous=True)
        rospy.Subscriber(self.input_depth_image_topic, Float32MultiArray, self.range_image_callback)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)
        rospy.spin()

    def range_image_callback(self, data):
        self.range_image = ((np.clip(np.array(data.data).reshape((64, 64, 1)), 0, 15) / 15.) * 255).astype(np.uint8)
        observation = {'depth_cam': np.copy(self.range_image), 'moving_target': np.array([5, 0, 0])}
        action = self.end_to_end_planner.infer(observation, deterministic=True)
        # publish the action

    def pose_callback(self, data):
        self.current_position = data.pose.position


if __name__ == '__main__':
    end_to_end_planner_node = EndToEndPlannerNode()
    end_to_end_planner_node.listen()
