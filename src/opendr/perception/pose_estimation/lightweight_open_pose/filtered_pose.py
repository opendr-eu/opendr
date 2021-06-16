# Copyright 2020-2021 OpenDR European Project
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

from opendr.engine.target import Pose
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.one_euro_filter import OneEuroFilter


class FilteredPose(Pose):
    def __init__(self, keypoints, confidence):
        super().__init__(keypoints, confidence)
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(self.num_kpts)]
