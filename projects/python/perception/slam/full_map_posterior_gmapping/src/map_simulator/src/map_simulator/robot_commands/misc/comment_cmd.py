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

from .misc_cmd import MiscCommand


class CommentCommand(MiscCommand):
    """
    Class for having comments in the JSON files, and while we are at it, be able to print them during simulation.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate Comment command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "txt": (string) (Optional)[Default: "") Comment
                                  * "print": (bool) (Optional)[Default: True) Print the contents of "txt" during
                                                    simulation if True.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command. Unused.
        """

        super(CommentCommand, self).__init__(config, callback, last_pose)

        if 'print' in config:
            do_print = bool(config['print'])
        else:
            do_print = True

        if 'txt' in config:
            self.msg = config['txt']
        else:
            self.msg = ""

        if self.msg and do_print:
            self.do_print = True
        else:
            self.do_print = False
