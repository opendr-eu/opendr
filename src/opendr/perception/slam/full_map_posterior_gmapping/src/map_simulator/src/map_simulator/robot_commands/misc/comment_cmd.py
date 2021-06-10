from misc_cmd import MiscCommand


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
