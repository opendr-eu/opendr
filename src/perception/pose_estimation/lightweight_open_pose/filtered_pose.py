from engine.target import Pose
from perception.pose_estimation.lightweight_open_pose.algorithm.modules.one_euro_filter import OneEuroFilter


class FilteredPose(Pose):
    def __init__(self, keypoints, confidence):
        super().__init__(keypoints, confidence)
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(self.num_kpts)]
