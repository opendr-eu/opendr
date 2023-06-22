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

from misc_cmd import MiscCommand


class SleepCommand(MiscCommand):
    """
    Class for letting time pass...
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate Scan command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "duration": (float) (Optional)[Default: 1) Time in seconds for doing nothing.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command. Unused.
        """

        super(SleepCommand, self).__init__(config, callback, last_pose)

        if 'duration' in config:
            self.ms = int(config['duration']) * 1000
        else:
            self.ms = 1000
