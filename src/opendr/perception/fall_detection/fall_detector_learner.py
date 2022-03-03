# Copyright 2020-2022 OpenDR European Project
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

from opendr.engine.learners import Learner
from opendr.engine.target import Category, Keypoint
from numpy import arccos, dot, rad2deg, linalg


class FallDetectorLearner(Learner):
    def __init__(self, pose_estimator):
        super().__init__()

        self.pose_estimator = pose_estimator

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        pass

    def eval(self, dataset):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def optimize(self, target_device):
        pass

    def reset(self):
        pass

    def infer(self, img):
        poses = self.pose_estimator.infer(img)
        if len(poses) != 0:
            return self.naive_fall_detection(poses[0])
        else:
            return Category(-1), [Keypoint([-1, -1]), Keypoint([-1, -1]), Keypoint([-1, -1])]

    @staticmethod
    def get_angle_to_horizontal(v1, v2):
        vector = abs(v1 - v2)
        unit_vector = vector / linalg.norm(vector)
        return rad2deg(arccos(dot(unit_vector, [1, 0])))

    def naive_fall_detection(self, pose):
        """
        This naive implementation of fall detection first establishes the average point between the two hips keypoints.
        It then tries to figure out the average position of the head and legs. Together with the hips point,
        two vectors, head-hips (torso) and hips-legs, are created, which give a general sense of the "verticality" of
        the body.
        """
        # Hip detection, hip average serves as the middle point of the pose
        if pose["r_hip"][0] != -1 and pose["l_hip"][0] != -1:
            hips = (pose["r_hip"] + pose["l_hip"])/2
        elif pose["r_hip"][0] != -1:
            hips = pose["r_hip"]
        elif pose["l_hip"][0] != -1:
            hips = pose["l_hip"]
        else:
            # Can't detect fall without hips
            return Category(-1), [Keypoint([-1, -1]), Keypoint([-1, -1]), Keypoint([-1, -1])]

        # Figure out head average position
        head = [-1, -1]
        # Try to detect approximate head position from shoulders, eyes, neck
        if pose["r_eye"][0] != -1 and pose["l_eye"][0] != -1 and pose["neck"][0] != -1:  # Eyes and neck detected
            head = (pose["r_eye"] + pose["l_eye"] + pose["neck"])/3
        elif pose["r_eye"][0] != -1 and pose["l_eye"][0] != -1:  # Eyes detected
            head = (pose["r_eye"] + pose["l_eye"]) / 2
        elif pose["r_sho"][0] != -1 and pose["l_sho"][0] != -1:  # Shoulders detected
            head = (pose["r_sho"] + pose["l_sho"]) / 2
        elif pose["neck"][0] != -1:  # Neck detected
            head = pose["neck"]

        # Figure out legs average position
        knees = [-1, -1]
        # Knees detection
        if pose["r_knee"][0] != -1 and pose["l_knee"][0] != -1:
            knees = (pose["r_knee"] + pose["l_knee"]) / 2
        elif pose["r_knee"][0] != -1:
            knees = pose["r_knee"]
        elif pose["l_knee"][0] != -1:
            knees = pose["l_knee"]
        ankles = [-1, -1]
        # Ankle detection
        if pose["r_ank"][0] != -1 and pose["l_ank"][0] != -1:
            ankles = (pose["r_ank"] + pose["l_ank"]) / 2
        elif pose["r_ank"][0] != -1:
            ankles = pose["r_ank"]
        elif pose["l_ank"][0] != -1:
            ankles = pose["l_ank"]

        legs = [-1, -1]
        if knees[0] != -1 and knees[1] != -1 and ankles[0] != -1 and ankles[1] != -1:
            legs = (knees + ankles) / 2
        elif ankles[0] != -1 and ankles[1] != -1:
            legs = ankles
        elif knees[0] != -1 and knees[1] != -1:
            legs = knees

        torso_vertical = -1
        # Figure out the head-hips vector (torso) angle to horizontal axis
        if head[0] != -1 and head[1] != -1:
            angle_to_horizontal = self.get_angle_to_horizontal(head, hips)
            if 45 < angle_to_horizontal < 135:
                torso_vertical = 1
            else:
                torso_vertical = 0

        legs_vertical = -1
        # Figure out the hips-legs vector angle to horizontal axis
        if legs[0] != -1 and legs[1] != -1:
            angle_to_horizontal = self.get_angle_to_horizontal(hips, legs)
            if 30 < angle_to_horizontal < 150:
                legs_vertical = 1
            else:
                legs_vertical = 0

        if legs_vertical != -1:
            if legs_vertical == 0:  # Legs are not vertical, probably not under torso, so person has fallen
                return Category(1), [Keypoint(head), Keypoint(hips), Keypoint(legs)]
            elif legs_vertical == 1:  # Legs are vertical, so person is standing
                return Category(0), [Keypoint(head), Keypoint(hips), Keypoint(legs)]
        elif torso_vertical != -1:
            if torso_vertical == 0:  # Torso is not vertical, without legs we assume person has fallen
                return Category(1), [Keypoint(head), Keypoint(hips), Keypoint(legs)]
            elif torso_vertical == 1:  # Torso is vertical, without legs we assume person is standing
                return Category(0), [Keypoint(head), Keypoint(hips), Keypoint(legs)]
        else:
            # Only hips detected, can't detect fall
            return Category(-1), [Keypoint([-1, -1]), Keypoint([-1, -1]), Keypoint([-1, -1])]
