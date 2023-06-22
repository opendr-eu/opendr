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

from std_msgs.msg import Bool as BoolMsg

from .message_cmd import MessageCommand


class BoolMessageCommand(MessageCommand):
    """
    Class for sending a boolean message to the SLAM node to let it know to start the localization only phase, so as to
    evaluate the performance of the algorithm.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate Localization-Only message command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "enable": (bool) (Optional)[Default: True) Send the localization-only
                                                     message if True.
                                  * "print": (bool) (Optional)[Default: True) Print a text during simulation if True to
                                                    let the user know that the message was sent.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command. Unused.
        """

        super(BoolMessageCommand, self).__init__(config, callback, last_pose)

        try:
            self.topic = config['topic']
        except KeyError:
            raise KeyError("Message topic not defined in configuration.")

        if 'value' in config:
            self.value = bool(config['enable'])
        else:
            self.value = True

        if 'desc' in config:
            self.do_print = True
            self.desc = config['desc']
        else:
            self.do_print = False
            self.desc = ""

        msg = BoolMsg()
        msg.data = self.value

        self.msg = msg
