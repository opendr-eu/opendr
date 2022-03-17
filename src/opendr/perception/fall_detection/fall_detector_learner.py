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

from os import walk, path
from csv import reader
from tqdm import tqdm

from numpy import arccos, dot, linalg, rad2deg

from opendr.engine.learners import Learner
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from opendr.engine.target import Category, Keypoint
from opendr.engine.data import Image


class FallDetectorLearner(Learner):
    def __init__(self, pose_estimator):
        super().__init__()

        self.pose_estimator = pose_estimator

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        pass

    def eval(self, dataset):
        """
        Evaluation on UR Fall Dataset, discards all temporary poses, then tries to detect the pose (note that in this
        dataset there is always one pose in the frame). If a pose a detected, fall detection is run on it and
        the result is compared to the label to get the carious metrics.

        The overall regular accuracy is reported, as well as sensitivity and specificity, the detection accuracy of the
        pose estimator, i.e. for how many poses we got a fall detection result, not counting the temporary poses.
        Lastly, it returns the absolute number of frames where the pose detection was entirely missed.

        """
        data = self.__prepare_val_dataset(dataset)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        no_detections = 0
        temp_poses = 0

        p_bar_desc = "Evaluation progress"
        pbar_eval = tqdm(desc=p_bar_desc, total=len(data), bar_format="{l_bar}%s{bar}{r_bar}" % '\x1b[38;5;231m')

        for i in range(len(data)):
            image = data[i][0]
            label = data[i][1].data
            if label == 0:  # Temporary pose, skip
                temp_poses += 1
                pbar_eval.update(1)
                continue

            image = Image.open(image)
            detections = self.infer(image)
            if len(detections) > 0:
                fallen = detections[0][0].data
            else:  # Can't detect fallen or standing
                no_detections += 1
                pbar_eval.update(1)
                continue

            if label == -1 and fallen == -1:  # Person standing, detected standing, true negative
                tn += 1
            if label == 1 and fallen == 1:  # Person fallen, detected fall, true positive
                tp += 1
            elif label == -1 and fallen == 1:  # Person standing, detected fall, false positive
                fp += 1
            elif label == 1 and fallen == -1:  # Person fallen, detected standing, false negative
                fn += 1
            pbar_eval.update(1)
        pbar_eval.close()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        detection_accuracy = (tp + tn + fp + fn) / (len(data) - temp_poses)

        return {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
                "detection_accuracy": detection_accuracy, "no_detections": no_detections}

    @staticmethod
    def __prepare_val_dataset(dataset):
        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() != "ur_fall_dataset":
                raise UserWarning("dataset_type must be \"ur_fall_dataset\"")

        # Get files and subdirectories of dataset.path directory
        f = []
        dirs = []
        for (dirpath, dirnames, filenames) in walk(dataset.path):
            f = filenames
            dirs = dirnames
            break

        # Verify csv files with labels are present
        if "urfall-cam0-adls.csv" not in f:
            raise UserWarning("Didn't find \"urfall-cam0-adls.csv\" file in the dataset path provided.")
        if "urfall-cam0-falls.csv" not in f:
            raise UserWarning("Didn't find \"urfall-cam0-falls.csv\" file in the dataset path provided.")

        # Verify all subfolders with images are present
        for i in range(1, 41, 1):
            dirname = f"adl-{i:02d}-cam0-rgb"
            if dirname not in dirs:
                raise UserWarning("Didn't find \"" + dirname + "\" dir in the dataset path provided.")

        for i in range(1, 31, 1):
            dirname = f"fall-{i:02d}-cam0-rgb"
            if dirname not in dirs:
                raise UserWarning("Didn't find \"" + dirname + "\" dir in the dataset path provided.")

        data = []
        labels = []
        with open(path.join(dataset.path, "urfall-cam0-adls.csv")) as adls_labels_file, \
                open(path.join(dataset.path, "urfall-cam0-falls.csv")) as falls_labels_file:
            # pass the file object to reader() to get the reader object
            adls_reader = reader(adls_labels_file)
            falls_reader = reader(falls_labels_file)
            # Iterate over each row in the csv using reader object
            for adls_row in adls_reader:
                # Append image path based on video id, frame id
                dataset_folder_name = "UR Fall Dataset"
                folder_name = adls_row[0] + "-cam0-rgb"
                img_name = folder_name + f"-{int(adls_row[1]):03d}.png"
                data.append(path.join(dataset_folder_name, folder_name, img_name))
                # Append label
                labels.append(Category(int(adls_row[2])))

            for falls_row in falls_reader:
                # Append image path based on video id, frame id
                folder_name = falls_row[0] + "-cam0-rgb"
                img_name = folder_name + f"-{int(falls_row[1]):03d}.png"
                data.append(path.join(dataset_folder_name, folder_name, img_name))
                # Append label
                labels.append(Category(int(falls_row[2])))

        return URFallDatasetIterator(data, labels)

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
        results = []
        for pose in poses:
            results.append(self.naive_fall_detection(pose))

        if len(results) >= 1:
            return results

        return []

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
            return Category(0), [Keypoint([-1, -1]), Keypoint([-1, -1]), Keypoint([-1, -1])], pose

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
                return Category(1), [Keypoint(head), Keypoint(hips), Keypoint(legs)], pose
            elif legs_vertical == 1:  # Legs are vertical, so person is standing
                return Category(-1), [Keypoint(head), Keypoint(hips), Keypoint(legs)], pose
        elif torso_vertical != -1:
            if torso_vertical == 0:  # Torso is not vertical, without legs we assume person has fallen
                return Category(1), [Keypoint(head), Keypoint(hips), Keypoint(legs)], pose
            elif torso_vertical == 1:  # Torso is vertical, without legs we assume person is standing
                return Category(-1), [Keypoint(head), Keypoint(hips), Keypoint(legs)], pose
        else:
            # Only hips detected, can't detect fall
            return Category(0), [Keypoint([-1, -1]), Keypoint([-1, -1]), Keypoint([-1, -1])], pose


class URFallDatasetIterator(DatasetIterator):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
