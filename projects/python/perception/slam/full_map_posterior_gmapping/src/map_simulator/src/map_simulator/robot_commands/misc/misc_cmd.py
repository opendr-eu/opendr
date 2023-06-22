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


class MiscCommand(Command):
    """
    Abstract class for creating simulated robot miscellaneous commands, such as scans, comments, messages, etc.
    Defines the required interfaces for miscellaneous commands.
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
        :param last_pose: (Pose) Last pose of the robot before this command. Mostly unused for miscellaneous commands.
        """

        super(MiscCommand, self).__init__(config, callback, last_pose)
