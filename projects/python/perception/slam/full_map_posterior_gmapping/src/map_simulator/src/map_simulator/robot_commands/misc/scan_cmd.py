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


class ScanCommand(MiscCommand):
    """
    Class for initiating a scan of the environment on command, apart from the automatic ones done after
    the move commands.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate Scan command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "scans": (int) (Optional)[Default: 1) Number of scans to perform.
                                  * "deterministic": (bool) (Optional)[Default: None] Overrides general deterministic
                                                     configuration for this cmd. If None, then general config applies.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command. Unused.
        """

        super(ScanCommand, self).__init__(config, callback, last_pose)

        if 'scans' in config:
            self.scans = int(config['scans'])
        else:
            self.scans = 1

        self._deterministic = None
        if "deterministic" in config:
            det = config["deterministic"]
            if not isinstance(det, bool):
                raise TypeError("Invalid type ({}: {}). Deterministic must be a boolean.".format(type(det), det))

            self._deterministic = det

    def get_deterministic(self):
        return self._deterministic
