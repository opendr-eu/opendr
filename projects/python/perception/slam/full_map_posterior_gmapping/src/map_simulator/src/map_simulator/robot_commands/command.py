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


class Command:
    """
    Abstract class for creating simulated robot commands, such as movements, scans, or other.
    Defines the required interfaces for commands.
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
        :param last_pose: (Pose) Last pose of the robot before this command. Mostly used for movement commands,
                                 but required for all types of commands for consistency purposes during instantiation.
        """

        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")

        if not callable(callback):
            raise AttributeError("Callback must be a function or other callable object.")

        self._callback = callback
        self._last_pose = last_pose
        self._poses = []

    def get_poses(self):
        """
        Returns the command's poses. If the command is not of the movement type, then it is an empty list.
        DON'T OVERRIDE! unless you know what you are doing.

        :return: (list) List of command's poses. Empty list if command is not movement.
        """

        return self._poses

    def __call__(self):
        """
        Executes the callback function with itself as an argument, so that the command's properties and methods are
        accessible from the callback.
        DON'T OVERRIDE! unless you know what you are doing.

        :return: () Whatever the callback function may return.
        """

        self._callback(self)
