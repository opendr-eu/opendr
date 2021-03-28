# Copyright 2020 Aristotle University of Thessaloniki
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

import numpy as np
from typing import List


class BaseTarget:
    """
    Root BaseTarget class has been created to allow for setting the hierarchy of different targets.
    Classes that inherit from BaseTarget can be used either as outputs of an algorithm or as ground
    truth annotations, but there is no guarantee that this is always possible, i.e. that both options are possible.

    Classes that are only used either for ground truth annotations or algorithm outputs must inherit this class.
    """
    def __init__(self):
        pass


class Target(BaseTarget):
    """
    Classes inheriting from the Target class always guarantee that they can be used for both cases, outputs and
    ground truth annotations.
    Therefore, classes that are only used to provide ground truth annotations
    must inherit from BaseTarget instead of Target. To allow representing different types of
    targets, this class serves as the basis for the more specialized forms of targets.
    All the classes should implement the corresponding setter/getter functions to ensure that the necessary
    type checking is performed (if there is no other technical obstacle to this, e.g., negative performance impact).
    """
    def __init__(self):
        super().__init__()
        self.data = None
        self.confidence = None
        self.action = None


class Keypoint(Target):
    """
    This target is used for keypoint detection in pose estimation, body part detection, etc.
    A keypoint is a list with two coordinates [x, y], which gives the x, y position of the
    keypoints on the image.
    """
    def __init__(self, keypoint, confidence=None):
        super().__init__()
        self.data = keypoint
        self.confidence = confidence

    def __str__(self):
        return str(self.data)


class Pose(Target):
    """
    This target is used for pose estimation. It contains a list of Keypoints.
    Refer to kpt_names for keypoint naming.
    """
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    last_id = -1

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.data = keypoints
        self.confidence = confidence
        self.id = None

    def __str__(self):
        """Matches kpt_names and keypoints x,y to get the best human-readable format for pose."""

        out_string = ""
        # noinspection PyUnresolvedReferences
        for name, kpt in zip(Pose.kpt_names, self.data.tolist()):
            out_string += name + ": " + str(kpt) + "\n"
        return out_string


class BoundingBox3D(Target):
    """
    This target is used for 3D Object Detection and Tracking.
    A bounding box is described by its location (x, y, z), dimensions (w, h, d) and rotation (along vertical y axis).
    Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
    truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
    observation angle of an object (alpha).
    """
    def __init__(
        self,
        name,
        truncated,
        occluded,
        alpha,
        bbox2d,
        dimensions,
        location,
        rotation_y,
        score=0,
    ):
        super().__init__()
        self.data = {
            "name": name,
            "truncated": truncated,
            "occluded": occluded,
            "alpha": alpha,
            "bbox2d": bbox2d,
            "dimensions": dimensions,
            "location": location,
            "rotation_y": rotation_y,
        }
        self.confidence = score

    def kitti(self):

        result = {}

        result["name"] = np.array([self.data["name"]])
        result["truncated"] = np.array([self.data["truncated"]])
        result["occluded"] = np.array([self.data["occluded"]])
        result["alpha"] = np.array([self.data["alpha"]])
        result["bbox"] = np.array([self.data["bbox2d"]])
        result["dimensions"] = np.array([self.data["dimensions"]])
        result["location"] = np.array([self.data["location"]])
        result["rotation_y"] = np.array([self.data["rotation_y"]])
        result["score"] = np.array([self.confidence])

        return result

    @property
    def name(self):
        return self.data["name"]

    @property
    def truncated(self):
        return self.data["truncated"]

    @property
    def occluded(self):
        return self.data["occluded"]

    @property
    def alpha(self):
        return self.data["alpha"]

    @property
    def bbox2d(self):
        return self.data["bbox2d"]

    @property
    def dimensions(self):
        return self.data["dimensions"]

    @property
    def location(self):
        return self.data["location"]

    @property
    def rotation_y(self):
        return self.data["rotation_y"]

    def __repr__(self):
        return "BoundingBox3D " + str(self)

    def __str__(self):
        return str(self.kitti())


class BoundingBox3DList(Target):
    """
    This target is used for 3D Object Detection. It contains a list of BoundingBox3D targets.
    A bounding box is described by its location (x, y, z), dimensions (l, h, w) and rotation (along vertical (y) axis).
    Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
    truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
    observation angle of an object (alpha).
    """
    def __init__(
        self,
        bounding_boxes_3d
    ):
        super().__init__()
        self.data: List[BoundingBox3D] = bounding_boxes_3d
        self.confidence = np.mean([box.confidence for box in self.data])

    @staticmethod
    def from_kitti(boxes_kitti):

        count = len(boxes_kitti["name"])

        boxes3d = []

        for i in range(count):
            box3d = BoundingBox3D(
                boxes_kitti["name"][i],
                boxes_kitti["truncated"][i],
                boxes_kitti["occluded"][i],
                boxes_kitti["alpha"][i],
                boxes_kitti["bbox"][i],
                boxes_kitti["dimensions"][i],
                boxes_kitti["location"][i],
                boxes_kitti["rotation_y"][i],
                boxes_kitti["score"][i],
            )

            boxes3d.append(box3d)

        return BoundingBox3DList(boxes3d)

    def kitti(self):

        result = {
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
            "score": [],
        }

        if len(self.data) == 0:
            return result
        elif len(self.data) == 1:
            return self.data[0].kitti()
        else:

            for box in self.data:
                result["name"].append(box.data["name"])
                result["truncated"].append(box.data["truncated"])
                result["occluded"].append(box.data["occluded"])
                result["alpha"].append(box.data["alpha"])
                result["bbox"].append(box.data["bbox2d"])
                result["dimensions"].append(box.data["dimensions"])
                result["location"].append(box.data["location"])
                result["rotation_y"].append(box.data["rotation_y"])
                result["score"].append(box.confidence)

            result["name"] = np.array(result["name"])
            result["truncated"] = np.array(result["truncated"])
            result["occluded"] = np.array(result["occluded"])
            result["alpha"] = np.array(result["alpha"])
            result["bbox"] = np.array(result["bbox"])
            result["dimensions"] = np.array(result["dimensions"])
            result["location"] = np.array(result["location"])
            result["rotation_y"] = np.array(result["rotation_y"])
            result["score"] = np.array(result["score"])

        return result

    @property
    def boxes(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "BoundingBox3DList " + str(self)

    def __str__(self):
        return str(self.kitti())


class TrackingBoundingBox3D(Target):
    """
    This target is used for 3D Object Tracking.
    A tracking bounding box is described by frame, id, its location (x, y, z),
    dimensions (w, h, d) and rotation (along vertical y axis).
    Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
    truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
    observation angle of an object (alpha).
    """
    def __init__(
        self,
        name,
        truncated,
        occluded,
        alpha,
        bbox2d,
        dimensions,
        location,
        rotation_y,
        id,
        score=0,
        frame=-1,
    ):
        super().__init__()
        self.data = {
            "name": name,
            "truncated": truncated,
            "occluded": occluded,
            "alpha": alpha,
            "bbox2d": bbox2d,
            "dimensions": dimensions,
            "location": location,
            "id": id,
            "frame": frame,
        }
        self.confidence = score

    def kitti(self, with_tracking_info=True):

        result = {}

        result["name"] = np.array([self.data["name"]])
        result["truncated"] = np.array([self.data["truncated"]])
        result["occluded"] = np.array([self.data["occluded"]])
        result["alpha"] = np.array([self.data["alpha"]])
        result["bbox"] = np.array([self.data["bbox2d"]])
        result["dimensions"] = np.array([self.data["dimensions"]])
        result["location"] = np.array([self.data["location"]])
        result["rotation_y"] = np.array([self.data["rotation_y"]])
        result["score"] = np.array([self.confidence])

        if with_tracking_info:
            result["id"] = np.array([self.data["id"]])
            result["frame"] = np.array([self.data["frame"]])

        return result

    @property
    def name(self):
        return self.data["name"]

    @property
    def truncated(self):
        return self.data["truncated"]

    @property
    def occluded(self):
        return self.data["occluded"]

    @property
    def alpha(self):
        return self.data["alpha"]

    @property
    def bbox2d(self):
        return self.data["bbox2d"]

    @property
    def dimensions(self):
        return self.data["dimensions"]

    @property
    def location(self):
        return self.data["location"]

    @property
    def rotation_y(self):
        return self.data["rotation_y"]

    @property
    def frame(self):
        return self.data["frame"]

    @property
    def id(self):
        return self.data["id"]

    def bounding_box_3d(self):
        return BoundingBox3D(
            self.name, self.truncated, self.occluded, self.alpha,
            self.bbox2d, self.dimensions, self.location, self.rotation_y,
            self.confidence
        )

    def __repr__(self):
        return "TrackingBoundingBox3D " + str(self)

    def __str__(self):
        return str(self.kitti(True))


class TrackingBoundingBox3DList(Target):
    """
    This target is used for 3D Object Tracking. It contains a list of TrackingBoundingBox3D targets.
    A tracking bounding box is described by frame, id, its location (x, y, z),
    dimensions (l, h, w) and rotation (along vertical (y) axis).
    Additional fields are used to describe confidence (score), 2D projection of the box on camera image (bbox2d),
    truncation (truncated) and occlusion (occluded) levels, the name of an object (name) and
    observation angle of an object (alpha).
    """
    def __init__(
        self,
        tracking_bounding_boxes_3d
    ):
        super().__init__()
        self.data: List[TrackingBoundingBox3D] = tracking_bounding_boxes_3d
        self.confidence = np.mean([box.confidence for box in self.data])

    @staticmethod
    def from_kitti(boxes_kitti, ids, frames=None):

        count = len(boxes_kitti["name"])

        if frames is None:
            frames = [-1] * count

        tracking_boxes_3d = []

        for i in range(count):
            box3d = TrackingBoundingBox3D(
                boxes_kitti["name"][i],
                boxes_kitti["truncated"][i],
                boxes_kitti["occluded"][i],
                boxes_kitti["alpha"][i],
                boxes_kitti["bbox"][i],
                boxes_kitti["dimensions"][i],
                boxes_kitti["location"][i],
                boxes_kitti["rotation_y"][i],
                ids[count],
                boxes_kitti["score"][i],
                frames[count],
            )

            tracking_boxes_3d.append(box3d)

        return TrackingBoundingBox3DList(tracking_boxes_3d)

    def kitti(self, with_tracking_info=True):

        result = {
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
            "score": [],
        }

        if with_tracking_info:
            result["id"] = []
            result["frame"] = []

        if len(self.data) == 0:
            return result
        elif len(self.data) == 1:
            return self.data[0].kitti()
        else:

            for box in self.data:
                result["name"].append(box.data["name"])
                result["truncated"].append(box.data["truncated"])
                result["occluded"].append(box.data["occluded"])
                result["alpha"].append(box.data["alpha"])
                result["bbox"].append(box.data["bbox2d"])
                result["dimensions"].append(box.data["dimensions"])
                result["location"].append(box.data["location"])
                result["rotation_y"].append(box.data["rotation_y"])
                result["score"].append(box.confidence)

                if with_tracking_info:
                    result["id"].append(box.data["id"])
                    result["frame"].append(box.data["frame"])

            result["name"] = np.array(result["name"])
            result["truncated"] = np.array(result["truncated"])
            result["occluded"] = np.array(result["occluded"])
            result["alpha"] = np.array(result["alpha"])
            result["bbox"] = np.array(result["bbox"])
            result["dimensions"] = np.array(result["dimensions"])
            result["location"] = np.array(result["location"])
            result["rotation_y"] = np.array(result["rotation_y"])
            result["score"] = np.array(result["score"])

            if with_tracking_info:
                result["id"] = np.array(result["id"])
                result["frame"] = np.array(result["frame"])

        return result

    @property
    def boxes(self):
        return self.data

    def bounding_box_3d_list(self):
        return BoundingBox3DList([box.bounding_box_3d() for box in self.data])

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "TrackingBoundingBox3DList " + str(self)

    def __str__(self):
        return str(self.kitti(True))
