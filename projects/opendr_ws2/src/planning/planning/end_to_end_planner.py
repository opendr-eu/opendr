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

import rclpy
from rclpy.node import Node
from opendr.planning.end_to_end_planning import EndToEndPlanningRLLearner, AgiEnv
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
import os


class EndToEndPlannerNode(Node):
    def __init__(self):
        """
        Initialize the End to end planner ROS node and create an instance of the respective learner class.
        """
        super().__init__('end_to_end_planner')

    def _init_learner(self) -> bool:
        """
        Initialize learner.
        """
        self.env = AgiEnv()
        self.end_to_end_planner = EndToEndPlanningRLLearner(self.env)
        print(os.environ.get("OPENDR_HOME"))
        self.end_to_end_planner.load(
            os.environ.get("OPENDR_HOME") + '/src/opendr/planning/end_to_end_planning/pretrained_model/saved_model.zip')

    def _init_subscriber(self):
        """
        Subscribe to all relevant topics.
        """
        pass

    def _init_publisher(self):
        """
        Set up the publishers as requested by the user.
        """
        pass

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        self.get_logger().info('EfficientPS node started!')
        if self._init_learner():
            self._init_publisher()
            self._init_subscriber()

    def callback(self, data):
        """
        Predict the panoptic segmentation map from the input image and publish the results.
        :param data: Input image message
        :type data: sensor_msgs.msg.Image
        """
        pass


def main():
    rclpy.init()
    end_to_end_planner_node = EndToEndPlannerNode()
    end_to_end_planner_node.listen()
    rclpy.spin(end_to_end_planner_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    end_to_end_planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
