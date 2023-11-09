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

import numpy as np

from move_cmd import MoveCommand

from map_simulator.geometry.primitives import Pose


class MoveInterpolationCommand(MoveCommand):
    """
    Class for a command for moving from a given start pose, to a given destination pose,
    in a given number of steps, by linearly interpolating the position's x and y coordinates
    and the orientation's angle theta between start and end.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate an Interpolation move command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "start_position": (list|np.ndarray) (Optional)[Default: last_pose.position)
                                                                        Starting position [x, y] of the robot.
                                  * "start_orientation": (float|list|np.ndarray) (Optional)[Default: last_pose.orient.)
                                                                           Starting orientation theta of the robot.
                                  * "end_position": (list|np.ndarray) Ending position [x, y] of the robot.
                                  * "end_orientation": (float|list|np.ndarray) Ending orientation theta of the robot.
                                  * "steps": (int) Number of desired steps/poses for the movement
                                  * "deterministic": (bool) (Optional)[Default: None] Overrides general deterministic
                                                     configuration for this move. If None, then general config applies.
                                  * "scans": (int) (Optional)[Default: None] Overrides general number of scans per move
                                             configuration for this move. If None, then general config applies.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command.
        """

        super(MoveInterpolationCommand, self).__init__(config, callback, last_pose)

        self._start_pose = None
        self._end_pose = None
        self._remove_first_pose = None
        self._steps = None

        if 'start_position' in config:
            start_pos = config['start_position']
        else:
            start_pos = self._last_pose.position

        if 'start_orientation' in config:
            start_ori = config['start_orientation']
        else:
            start_ori = self._last_pose.orientation

        # If neither the start pose nor the start orientation were specified,
        # assume the user wanted to start where he left off and delete the
        # first generated pose from the list to avoid duplicate poses.
        remove_first_pose = False
        if 'start_position' not in config and 'start_orientation' not in config:
            remove_first_pose = True

        self.set_rm_first_pose(remove_first_pose, compute_poses=False)
        self.set_start_pose(Pose(start_pos, start_ori), compute_poses=False)
        self.set_end_pose(Pose(config['end_position'], config['end_orientation']), compute_poses=False)
        self.set_steps(config['steps'], compute_poses=True)

    def set_rm_first_pose(self, remove_first_pose, compute_poses=True):
        """
        Sets the remove_first_pose property, used to determine whether we want to move the robot to the starting pose
        or not. By default, if neither the starting position nor the starting orientation were explicitly defined, we
        assume that the user wants to continue with the last trajectory without jumping to a new pose.

        :param remove_first_pose: (bool) True if we want to ignore the starting pose.
                                         False if we want the robot to jump to the starting pose from wherever he was.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """

        self._remove_first_pose = remove_first_pose

        if compute_poses:
            self.compute_poses()

    def set_start_pose(self, pose, compute_poses=True):
        """
        Sets the starting pose of the robot.

        :param pose: (Pose) Starting pose of the robot.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """

        self._start_pose = pose

        if compute_poses:
            self.compute_poses()

    def set_end_pose(self, pose, compute_poses=True):
        """
        Sets the ending pose of the robot.

        :param pose: (Pose) Ending pose of the robot.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """

        self._end_pose = pose

        if compute_poses:
            self.compute_poses()

    def set_steps(self, steps, compute_poses=True):
        """
        Sets the number of poses to be interpolated.

        :param steps: (int) Number of desired poses/steps for the movement.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """

        self._steps = steps

        if compute_poses:
            self.compute_poses()

    def compute_poses(self):
        """
        Generates the movement's pose list and stores it internally.

        :return: (None)
        """

        if self._start_pose is None or \
                self._end_pose is None or \
                self._steps is None or \
                self._remove_first_pose is None:
            self._poses = []
            return

        steps = self._steps
        if self._remove_first_pose:
            steps += 1

        tmp_positions = np.linspace(self._start_pose.position, self._end_pose.position, num=steps)
        tmp_orientations = np.linspace(self._start_pose.orientation, self._end_pose.orientation, num=steps)

        tmp_poses = [Pose(tmp_positions[i], tmp_orientations[i]) for i in range(tmp_orientations.shape[0])]

        if self._remove_first_pose:
            tmp_poses = tmp_poses[1:]

        self._poses = tmp_poses
