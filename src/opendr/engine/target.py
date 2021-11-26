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

from abc import ABC
import numpy as np
from typing import Optional, Dict, Tuple, Any


class BaseTarget(ABC):
    """
    Root BaseTarget abstract class has been created to allow for setting the hierarchy of different targets.
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
        self._data = None
        self._confidence = None
        self._action = None

    @property
    def data(self):
        """
        Getter of data field.
        This returns the internal representation of the data.
        :return: the actual data held by the object
        :rtype: varies according to the actual concrete implementation
        """
        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data. This will perform the necessary type checking (if needed).
        :param: data to be assigned to the object
        """
        self._data = data

    @property
    def confidence(self):
        """
        Getter of confidence field.
        This returns the confidence for the current target.
        :return: the confidence held by the object
        :rtype: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        """
        Setter for the confidence field. This can be used to perform the necessary type checking (if needed).
        :param: confidence to be used for assigning confidence to this object
        """
        self._confidence = confidence

    @property
    def action(self):
        """
        Getter of action field.
        This returns the selected/expected action.
        :return: the action data held by the object
        :rtype: Action
        """
        return self._action

    @action.setter
    def action(self, action):
        """
        Setter for action. This will perform the necessary type checking (if needed).
        :param: action to be assigned to the object
        """
        self._action = action


class Category(Target):
    """
    The Category target is used for 1-of-K classification problems.
    It contains the predicted class or ground truth and optionally the description of the predicted class
    and the prediction confidence.
    """

    def __init__(self, prediction: int, description=None, confidence=None):
        """Initialize a category.

        Args:
            prediction (int): Class integer
            description (optional):
                Class description / translation of prediction to class name. Defaults to None
            confidence (optional):
                One-dimensional array / tensor of class probabilities. Defaults to None.
        """
        super().__init__()
        self._description = None

        if prediction is not None:
            self.data = prediction

        if description is not None:
            self.description = description

        if confidence is not None:
            self.confidence = confidence

    @property
    def description(self):
        """
        Getter of description field.
        This returns the description of the corresponding class.
        :return: the description of the corresponding class
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Setter for description.
        :param: description to be assigned for the winning class
        """
        if isinstance(description, str):
            self._description = description
        else:
            raise ValueError("Description should be a string")

    @property
    def data(self):
        """
        Getter of data.

        :return: the actual category held by the object
        :rtype: int
        """
        if self._data is None:
            raise ValueError("Category is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data. Category expects data of int type.
        :param: data to be used for creating a Category object
        """
        if isinstance(data, int):
            self._data = data
        else:
            raise ValueError("Category expects integers as data")

    def __str__(self):
        if self.description is None:
            if self.confidence is not None:
                return f"Class {self.data} with confidence {self.confidence}"
            else:
                return f"Class {self.data} "
        else:
            if self.confidence is not None:
                return f"Class {self.data} ({self.description}) with confidence {self.confidence}"
            else:
                return f"Class {self.data}, ({self.description})"


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
        self._id = None

    @property
    def id(self):
        """
        Getter of human id.

        :return: the actual human id held by the object
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Setter for human id to which the Pose corresponds to. Pose expects id to be of int type.
        Please note that None is a valid value, since a pose is not always accompanied with an id.
        :param: human id to which the Pose corresponds to
        """
        if isinstance(id, int) or id is None:
            self._id = id
        else:
            raise ValueError("Pose id should be an integer or None")

    @property
    def data(self):
        """
        Getter of data.

        :return: the actual pose data held by the object
        :rtype: numpy.ndarray
        """
        if self._data is None:
            raise ValueError("Pose object is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data. Pose expects a NumPy array or a list
        :param: data to be used for creating Pose
        """
        if isinstance(data, np.ndarray) or isinstance(data, list):
            self._data = data
        else:
            raise ValueError("Pose expects either NumPy arrays or lists as data")

    def __str__(self):
        """Matches kpt_names and keypoints x,y to get the best human-readable format for pose."""

        out_string = ""
        # noinspection PyUnresolvedReferences
        for name, kpt in zip(Pose.kpt_names, self.data.tolist()):
            out_string += name + ": " + str(kpt) + "\n"
        return out_string

    def __getitem__(self, key):
        """  Allows for accessing keypoint position using either integers or keypoint names """
        if isinstance(key, int):
            if key >= Pose.num_kpts or key < 0:
                raise ValueError('Pose supports ' + str(Pose.num_kpts) + ' keypoints. Keypoint id ' + str(
                    key) + ' is not within the supported range')
            else:
                return self.data[key]
        elif isinstance(key, str):
            try:
                position = Pose.kpt_names.index(key)
                return self.data[position]
            except:
                raise ValueError('Keypoint ' + key + ' not supported.')
        else:
            raise ValueError('Only string and integers are supported for retrieving keypoints.')


class BoundingBox(Target):
    """
    This target is used for 2D Object Detection.
    A bounding box is described by the left-top corner and its width and height.
    """
    def __init__(
        self,
        name,
        left,
        top,
        width,
        height,
        score=0,
    ):
        super().__init__()
        self.name = name
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.confidence = score

    def mot(self, with_confidence=True, frame=-1):

        if with_confidence:
            result = np.array([
                self.frame,
                self.left,
                self.top,
                self.width,
                self.height,
                self.confidence,
            ], dtype=np.float32)
        else:
            result = np.array([
                self.frame,
                self.left,
                self.top,
                self.width,
                self.height,
            ], dtype=np.float32)

        return result

    def coco(self, with_confidence=True):
        result = {}
        result['bbox'] = [self.left, self.top, self.width, self.height]
        result['category_id'] = self.name
        result['area'] = self.width * self.height
        return result

    def __repr__(self):
        return "BoundingBox " + str(self)

    def __str__(self):
        return str(self.mot())


class CocoBoundingBox(BoundingBox):
    """
    This target is used for 2D Object Detection with COCO format targets.
    A bounding box is described by the left-top corner and its width and height.
    Also, a segmentation of the target is returned if available.
    """

    def __init__(
        self,
        name,
        left,
        top,
        width,
        height,
        segmentation=[],
        area=0,
        iscrowd=0,
        score=0,
    ):
        super().__init__(name=name, left=left, top=top, width=width,
                         height=height, score=score)
        self.segmentation = segmentation
        self.iscrowd = iscrowd
        self.area = area

    def coco(self, with_confidence=True):
        result = {}
        result['bbox'] = [self.left, self.top, self.width, self.height]
        result['category_id'] = self.name
        if len(self.segmentation) > 0:
            result['area'] = self.area
            result['segmentation'] = self.segmentation
            result['iscrowd'] = self.iscrowd
        else:
            result['area'] = self.width * self.height
        if with_confidence:
            result['confidence'] = self.confidence
        return result

    def __repr__(self):
        return "BoundingBox " + str(self)

    def __str__(self):
        return str(self.coco())


class BoundingBoxList(Target):
    """
    This target is used for 2D Object Detection.
    A bounding box is described by the left-top corner and its width and height.
    """
    def __init__(
        self,
        boxes,
    ):
        super().__init__()
        self.data = boxes
        self.confidence = np.mean([box.confidence for box in self.data])

    @staticmethod
    def from_coco(boxes_coco, image_id=0):
        count = len(boxes_coco)

        boxes = []
        for i in range(count):
            if 'segmentation' in boxes_coco[i]:
                segmentation = boxes_coco[i]['segmentation']
            if 'iscrowd' in boxes_coco[i]:
                iscrowd = boxes_coco[i]['iscrowd']
            else:
                iscrowd = 0
            if 'area' in boxes_coco[i]:
                area = boxes_coco[i]['area']
            else:
                area = boxes_coco[i]['bbox'][2] * boxes_coco[i]['bbox'][3]
            box = CocoBoundingBox(
                boxes_coco[i]['category_id'],
                boxes_coco[i]['bbox'][0],
                boxes_coco[i]['bbox'][1],
                boxes_coco[i]['bbox'][2],
                boxes_coco[i]['bbox'][3],
                segmentation=segmentation,
                iscrowd=iscrowd,
                area=area,
            )
            boxes.append(box)

        return BoundingBoxList(boxes, image_id=image_id)

    def mot(self, with_confidence=True):

        result = np.array([
            box.mot(with_confidence) for box in self.data
        ])

        return result

    @property
    def boxes(self):
        return self.data

    def __getitem__(self, idx):
        return self.boxes[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "BoundingBoxList " + str(self)

    def __str__(self):
        return str(self.mot())


class TrackingAnnotation(Target):
    """
    This target is used for 2D Object Tracking.
    A tracking bounding box is described by id, the left-top corner and its width and height.
    """
    def __init__(
        self,
        name,
        left,
        top,
        width,
        height,
        id,
        score=0,
        frame=-1,
    ):
        super().__init__()
        self.name = name
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.id = id
        self.confidence = score
        self.frame = frame

    @staticmethod
    def from_mot(data):
        return TrackingAnnotation(
            0,
            data[2],
            data[3],
            data[4],
            data[5],
            data[1],
            data[6] if len(data) > 6 else 1,
            data[0],
        )

    def mot(self, with_confidence=True):

        if with_confidence:
            result = np.array([
                self.frame,
                self.id,
                self.left,
                self.top,
                self.width,
                self.height,
                self.confidence,
            ], dtype=np.float32)
        else:
            result = np.array([
                self.frame,
                self.id,
                self.left,
                self.top,
                self.width,
                self.height,
            ], dtype=np.float32)

        return result

    def bounding_box(self):
        return BoundingBox(self.name, self.left, self.top, self.width, self.height, self.confidence)

    def __repr__(self):
        return "TrackingAnnotation " + str(self)

    def __str__(self):
        return str(self.mot())


class TrackingAnnotationList(Target):
    """
    This target is used for 2D Object Tracking.
    A bounding box is described by the left and top corners and its width and height.
    """
    def __init__(
        self,
        boxes,
    ):
        super().__init__()
        self.data = boxes
        self.confidence = np.mean([box.confidence for box in self.data])

    @staticmethod
    def from_mot(data):
        boxes = []
        for box in data:
            boxes.append(TrackingAnnotation.from_mot(box))

        return TrackingAnnotationList(boxes)

    def mot(self, with_confidence=True):

        result = np.array([
            box.mot(with_confidence) for box in self.data
        ])

        return result

    def bounding_box_list(self):
        return BoundingBoxList([box.bounding_box() for box in self.data])

    @property
    def boxes(self):
        return self.data

    def __getitem__(self, idx):
        return self.boxes[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "TrackingAnnotationList " + str(self)

    def __str__(self):
        return str(self.mot())


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
        num_ground_truths = 1
        num_objects = 1
        index = list(range(num_objects)) + [-1] * (num_ground_truths - num_objects)
        result["index"] = np.array(index, dtype=np.int32)
        result["group_ids"] = np.arange(num_ground_truths, dtype=np.int32)

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
        self.data = bounding_boxes_3d
        self.confidence = None if len(self.data) == 0 else np.mean([box.confidence for box in self.data])

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

            num_ground_truths = len(result["name"])
            num_objects = len([x for x in result["name"] if x != "DontCare"])
            index = list(range(num_objects)) + [-1] * (num_ground_truths - num_objects)
            result["index"] = np.array(index, dtype=np.int32)
            result["group_ids"] = np.arange(num_ground_truths, dtype=np.int32)

        return result

    @property
    def boxes(self):
        return self.data

    def __getitem__(self, idx):
        return self.boxes[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "BoundingBox3DList " + str(self)

    def __str__(self):
        return str(self.kitti())


class TrackingAnnotation3D(BoundingBox3D):
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
        self.data = {
            "name": name,
            "truncated": truncated,
            "occluded": occluded,
            "alpha": alpha,
            "bbox2d": bbox2d,
            "dimensions": dimensions,
            "location": location,
            "rotation_y": rotation_y,
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
        return "TrackingAnnotation3D " + str(self)

    def __str__(self):
        return str(self.kitti(True))


class TrackingAnnotation3DList(Target):
    """
    This target is used for 3D Object Tracking. It contains a list of TrackingAnnotation3D targets.
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
        self.data = tracking_bounding_boxes_3d
        self.confidence = None if len(self.data) == 0 else np.mean([box.confidence for box in self.data])

    @staticmethod
    def from_kitti(boxes_kitti, ids, frames=None):

        count = len(boxes_kitti["name"])

        if frames is None:
            frames = [-1] * count

        tracking_boxes_3d = []

        for i in range(count):
            box3d = TrackingAnnotation3D(
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

        return TrackingAnnotation3DList(tracking_boxes_3d)

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

    def __getitem__(self, idx):
        return self.boxes[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "TrackingAnnotation3DList " + str(self)

    def __str__(self):
        return str(self.kitti(True))


class Heatmap(Target):
    """
    This target is used for multi-class segmentation problems or multi-class problems that require heatmap annotations.

    The data has to be a NumPy array.
    The attribute 'class_names' can be used to store a mapping from the numerical labels to string representations.
    """

    def __init__(self,
                 data: np.ndarray,
                 class_names: Optional[Dict[Any, str]]=None):
        super().__init__()
        self._class_names = None

        self.data = data
        if class_names is not None:
            self.class_names = class_names

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            raise ValueError('Data is empty.')
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError('Data must be a numpy array.')
        self._data = data

    @property
    def class_names(self) -> Dict[Any, str]:
        return self._class_names

    @class_names.setter
    def class_names(self, class_names: Dict[Any, str]):
        if not isinstance(class_names, dict):
            raise TypeError('Class_names must be a dictionary.')
        for key, value in class_names.items():
            if not isinstance(value, str):
                raise TypeError('Values of class_names must be string.')
        self._class_names = class_names

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.
        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data.
        return self.data

    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the underlying NumPy array.
        :return: shape of the data object
        :rtype: tuple of integers
        """
        return self.data.shape

    def __str__(self) -> str:
        """
        Returns a human-friendly string-based representation of the data.
        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return str(self.data)
