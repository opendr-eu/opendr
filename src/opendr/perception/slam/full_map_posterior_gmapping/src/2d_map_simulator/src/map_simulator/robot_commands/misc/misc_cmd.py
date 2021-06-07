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
