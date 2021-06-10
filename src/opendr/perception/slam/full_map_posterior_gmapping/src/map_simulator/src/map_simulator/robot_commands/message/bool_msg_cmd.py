from std_msgs.msg import Bool as BoolMsg

from message_cmd import MessageCommand


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
