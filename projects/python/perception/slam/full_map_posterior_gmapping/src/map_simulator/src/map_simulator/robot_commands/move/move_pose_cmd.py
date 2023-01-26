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

from move_cmd import MoveCommand
from map_simulator.geometry.primitives import Pose


class MovePoseCommand(MoveCommand):
    """
    Class for a command for moving directly to a given position and orientation.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate a Pose move command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "position": (list|np.ndarray) Target position [x, y] of the robot.
                                  * "orientation": (float|list|np.ndarray) Target orientation theta of the robot.
                                  * "deterministic": (bool) (Optional)[Default: None] Overrides general deterministic
                                                     configuration for this move. If None, then general config applies.
                                  * "scans": (int) (Optional)[Default: None] Overrides general number of scans per move
                                             configuration for this move. If None, then general config applies.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command. Unused.
        """

        super(MovePoseCommand, self).__init__(config, callback, last_pose)

        self._pose = None

        self.set_pose(Pose(config['position'], config['orientation']))

    def set_pose(self, pose):
        """
        Sets the target pose of the robot, and generates the self._poses

        :param pose: (Pose) Target pose of the robot.

        :return: None
        """

        self._pose = pose

        self.compute_poses()

    def compute_poses(self):
        """
        Generates the movement's pose list and stores it internally.

        :return: (None)
        """
        if self._pose is None:
            self._poses = []
        else:
            self._poses = [self._pose]
