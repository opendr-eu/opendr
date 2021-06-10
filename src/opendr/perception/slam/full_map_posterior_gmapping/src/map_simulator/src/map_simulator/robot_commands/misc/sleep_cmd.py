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
