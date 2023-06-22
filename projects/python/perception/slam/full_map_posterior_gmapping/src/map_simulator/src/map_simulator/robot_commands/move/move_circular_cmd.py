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
from map_simulator.utils import to_np


class MoveCircularCommand(MoveCommand):
    """
    Class for a command for moving in a circular arch, from a given start position, to a given destination position,
    equidistant from a given center position, in a given number of steps,
    in equally distributed angular increments, and having the robot's orientation follow the tangent of the
    circular trajectory.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate an Interpolation move command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "start_position": (list|np.ndarray) (Optional)[Default: last_pose.position)
                                                                        Starting position [x, y] of the robot.
                                  * "end_position": (list|np.ndarray) Ending position [x, y] of the robot.
                                  * "center": (list|np.ndarray) Position [x, y] of circular path's center.
                                  * "steps": (int) Number of desired steps/poses for the movement.
                                  * "dir": (string) Direction of turning. Can be:
                                      - "cw" : for clockwise rotation.
                                      - "ccw": for counter-clockwise rotation.
                                  * "deterministic": (bool) (Optional)[Default: None] Overrides general deterministic
                                                     configuration for this move. If None, then general config applies.
                                  * "scans": (int) (Optional)[Default: None] Overrides general number of scans per move
                                             configuration for this move. If None, then general config applies.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command.
        """

        super(MoveCircularCommand, self).__init__(config, callback, last_pose)

        self._start_position = None
        self._end_position = None
        self._center = None
        self._cw = None
        self._steps = None
        self._remove_first_pose = None

        if "start_position" in config:
            start_position = config['start_position']
            remove_first_pose = False
        else:
            start_position = self._last_pose.position
            remove_first_pose = True

        cw = True
        if "dir" in config:
            if config['dir'] == "ccw":
                cw = False

        self.set_start_position(start_position, compute_poses=False)
        self.set_end_position(config['end_position'], compute_poses=False)
        self.set_center(config['center'], compute_poses=False)
        self.set_steps(config['steps'], compute_poses=False)
        self.set_cw(cw, compute_poses=False)
        self.set_remove_first_pose(remove_first_pose, compute_poses=True)

    def set_start_position(self, position, compute_poses=True):
        """
        Sets the starting position of the robot.

        :param position: (list|np.ndarray) Starting position [x, y] of the robot.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """

        self._start_position = to_np(position)

        if compute_poses:
            self.compute_poses()

    def set_end_position(self, position, compute_poses=True):
        """
        Sets the target position of the robot.

        :param position: (list|np.ndarray) Target position [x, y] of the robot.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """

        self._end_position = to_np(position)

        if compute_poses:
            self.compute_poses()

    def set_center(self, position, compute_poses=True):
        """
        Sets the position of the circular trajectory's center.

        :param position: (list|np.ndarray) Center position [x, y] of the circular arch to follow.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """

        self._center = to_np(position)

        if compute_poses:
            self.compute_poses()

    def set_cw(self, cw, compute_poses=True):
        """
        Sets the direction of the rotation.

        :param cw: (bool) True if rotation is clockwise. False if counter-clockwise.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """
        self._cw = bool(cw)

        if compute_poses:
            self.compute_poses()

    def set_remove_first_pose(self, remove_first_pose, compute_poses=True):
        """
        Sets the remove_first_pose property, used to determine whether we want to move the robot to the starting pose
        or not. By default, if neither the starting position nor the starting orientation were explicitly defined, we
        assume that the user wants to continue with the last trajectory without jumping to a new pose.

        :param remove_first_pose: (bool) True if we want to ignore the starting pose.
                                         False if we want the robot to jump to the starting pose from wherever he was.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """
        self._remove_first_pose = bool(remove_first_pose)

        if compute_poses:
            self.compute_poses()

    def set_steps(self, steps, compute_poses=True):
        """
        Sets the number of poses to be interpolated.

        :param steps: (int) Number of desired poses/steps for the movement.
        :param compute_poses: (bool)[Default: True] Recompute the robot pose list if True.

        :return: (None)
        """

        self._steps = int(steps)

        if compute_poses:
            self.compute_poses()

    def compute_poses(self):
        """
        Generates the movement's pose list and stores it internally.

        :return: (None)
        """

        if self._start_position is None or \
                self._end_position is None or \
                self._center is None or \
                self._steps is None or \
                self._cw is None or \
                self._remove_first_pose is None:
            self._poses = []
            return

        start_diff = self._start_position - self._center
        start_angle = np.arctan2(start_diff[1], start_diff[0])
        end_diff = self._end_position - self._center
        end_angle = np.arctan2(end_diff[1], end_diff[0])

        if self._cw:
            while start_angle < end_angle:
                end_angle -= 2 * np.pi
        else:
            while start_angle > end_angle:
                end_angle += 2 * np.pi

        # Average the distances from start-center and end-center to estimate the radius for ease.
        # Hopefully the user entered a start and end position that are roughly equidistant from the center.
        radius = np.sqrt(np.dot(start_diff, start_diff))
        radius += np.sqrt(np.dot(end_diff, end_diff))
        radius /= 2

        steps = self._steps
        if self._remove_first_pose:
            steps += 1

        tmp_ori = np.linspace(start_angle, end_angle, num=steps)

        tmp_pos_x = self._center[0] + radius * np.cos(tmp_ori)
        tmp_pos_y = self._center[1] + radius * np.sin(tmp_ori)

        # Make orientations tangent to the circular trajectory
        if self._cw:
            tmp_ori -= np.pi / 2
        else:
            tmp_ori += np.pi / 2

        tmp_poses = [Pose([tmp_pos_x[i], tmp_pos_y[i]], tmp_ori[i]) for i in range(tmp_ori.shape[0])]

        if self._remove_first_pose:
            tmp_poses = tmp_poses[1:]

        self._poses = tmp_poses
