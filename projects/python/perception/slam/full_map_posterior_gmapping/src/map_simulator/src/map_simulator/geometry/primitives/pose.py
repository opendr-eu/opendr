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

from map_simulator.utils import to_np, normalize_angles


class Pose(object):

    def __init__(self, pose, orientation):
        self.position = to_np(pose)
        self.orientation = normalize_angles(to_np(orientation))

    def __str__(self):
        """
        String representation of the pose

        :return: (string) String representation of the pose.
        """

        pose_str = "[Pos: ("

        for coord in self.position:
            pose_str += str(coord) + ", "

        pose_str = pose_str[0:-2] + ") Orient: ("

        for coord in self.orientation:
            pose_str += str(coord) + ", "

        pose_str = pose_str[0:-2] + ")]"

        return pose_str

    def __repr__(self):
        """
        String representation of the pose

        :return: (string) String representation of the pose.
        """

        return self.__str__()
