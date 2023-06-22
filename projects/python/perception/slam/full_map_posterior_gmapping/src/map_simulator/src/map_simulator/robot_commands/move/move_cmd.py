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

from abc import ABCMeta, abstractmethod

from map_simulator.robot_commands.command import Command


class MoveCommand(Command):
    """
    Abstract class for creating simulated robot movement commands.
    Defines the required interfaces for movement commands.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, config, callback, last_pose):
        """
        Used for initializing the abstract class and also performing argument verifications for child classes.
        It sets the class' callback and last pose properties, so that they don't have to be done for every child class.

        :param config: (dict) Dictionary containing configuration of the command.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command. Used by some movement commands.
        """

        super(MoveCommand, self).__init__(config, callback, last_pose)

        # Properties for overriding the default general configuration for a given move
        self._deterministic = None
        self._scans = None

        if "deterministic" in config:
            det = config["deterministic"]
            if not isinstance(det, bool):
                raise TypeError("Invalid type ({}: {}). Deterministic must be a boolean.".format(type(det), det))

            self._deterministic = det

        if "scans" in config:
            scans = config["scans"]
            if not isinstance(scans, int):
                raise TypeError("Invalid type ({}: {}). Scans must be an integer.".format(type(scans), scans))
            if scans < 0:
                raise ValueError("Invalid value ({}). Number of scans must be a non-negative integer.".format(scans))

            self._scans = scans

    def get_meas_per_pose(self):
        return self._scans

    def get_deterministic(self):
        return self._deterministic

    @abstractmethod
    def compute_poses(self):
        """
        Method signature for generating the movement's pose list and stores it internally.

        :return: (None)
        """

        pass
