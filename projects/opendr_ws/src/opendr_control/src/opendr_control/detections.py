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


from math import sqrt
import rospy


class Detections:

    def __init__(self):
        self.objects = dict()
        self._distance_threshold = 0.01  # in m
        self._category_database = " "

    def process_detection(self, msg):
        if msg.id in self.objects:
            if (not self.object_already_stored(msg.id, msg.pose.pose)):
                self.objects[msg.id].append(msg.pose.pose)
        else:
            self.objects[msg.id] = [msg.pose.pose]

    def save_categories(self, msg):
        self._category_database = msg.database_location

    def find_object_by_category(self, category_name):
        '''
        Use the object's name to find its class id.
        :param category_name: name of the object
        :type category_name: str
        :return: class id
        :rtype: int
        '''
        if rospy.has_param(self._category_database):
            for key, item in rospy.get_param(self._category_database).items():
                if item == category_name:
                    return int(key)

    def get_object_pose(self, object_id):
        '''
        Use the class id to find the pose of one of the corresponding objects.
        :param object_id: class id of the objects to look fo
        :type category_name: int
        :return: object pose
        :rtype: geometry_msgs/Pose
        '''
        result = False
        if object_id in self.objects:
            result = self.objects[object_id][-1]
        return result

    def __calculate_distance(self, pose1, pose2):
        return sqrt((pose2.position.x - pose1.position.x)**2 + (pose2.position.y - pose1.position.y)**2)

    def object_already_stored(self, pred_class, pose1):
        result = False
        for pose2 in self.objects[pred_class]:
            if (self.__calculate_distance(pose1, pose2) < self._distance_threshold):
                result = True
                break
        return result
