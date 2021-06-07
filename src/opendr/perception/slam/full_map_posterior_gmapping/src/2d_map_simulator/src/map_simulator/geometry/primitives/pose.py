from map_simulator.utils import to_np, normalize_angles


class Pose(object):

    def __init__(self, pose, orientation):
        self.position = to_np(pose)
        self.orientation = normalize_angles(to_np(orientation))

    def __str__(self):
        """
        String representation of the pose

        :return: (string) String representation of the pose.
        """

        pose_str = "[Pos: ("

        for coord in self.position:
            pose_str += str(coord) + ", "

        pose_str = pose_str[0:-2] + ") Orient: ("

        for coord in self.orientation:
            pose_str += str(coord) + ", "

        pose_str = pose_str[0:-2] + ")]"

        return pose_str

    def __repr__(self):
        """
        String representation of the pose

        :return: (string) String representation of the pose.
        """

        return self.__str__()