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

from move_interpol_cmd import MoveInterpolationCommand

from map_simulator.utils import to_np


class MoveLinearCommand(MoveInterpolationCommand):
    """
    Class for a command for moving from a given start pose, to a given destination pose,
    in a given number of steps, by linearly interpolating the position's x and y coordinates
    between start and end, and having the robot's orientation follow the line defined by the
    start and end points.
    Works by computing the angle of the line between start and end positions, and feeding it
    to the Interpolation command in the configuration dictionary.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate a Linear move command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "start_position": (list|np.ndarray) (Optional)[Default: last_pose.position)
                                                                        Starting position [x, y] of the robot.
                                  * "end_position": (list|np.ndarray) Ending position [x, y] of the robot.
                                  * "steps": (int) Number of desired steps/poses for the movement
                                  * "deterministic": (bool) (Optional)[Default: None] Overrides general deterministic
                                                     configuration for this move. If None, then general config applies.
                                  * "scans": (int) (Optional)[Default: None] Overrides general number of scans per move
                                             configuration for this move. If None, then general config applies.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command.
        """

        if 'start_position' in config:
            start = to_np(config['start_position'])
        else:
            start = last_pose.position

        end = to_np(config['end_position'])

        diff = end - start
        theta = np.arctan2(diff[1], diff[0])
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        config['end_orientation'] = theta

        if 'start_position' in config or 'start_orientation' in config or abs(theta - last_pose.orientation) > 1e-6:
            config['start_orientation'] = theta

        super(MoveLinearCommand, self).__init__(config, callback, last_pose)
