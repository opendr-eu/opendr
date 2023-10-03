# Copyright 2020-2023 OpenDR European Project
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


from opendr.engine.target import BoundingBoxList
from opendr.perception.object_detection_2d.centernet.centernet_learner import CenterNetDetectorLearner
from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner
from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner
from opendr.perception.object_detection_2d.yolov3.yolov3_learner import YOLOv3DetectorLearner
from opendr.perception.object_detection_2d.yolov5.yolov5_learner import YOLOv5DetectorLearner
from opendr.perception.object_detection_2d.nanodet.nanodet_learner import NanodetLearner


class FilteredLearnerWrapper:
    def __init__(self, learner, allowed_classes=None):
        self.learner = learner
        self.allowed_classes = allowed_classes if allowed_classes is not None else []

        if isinstance(self.learner,
                      (CenterNetDetectorLearner, YOLOv3DetectorLearner, YOLOv5DetectorLearner, NanodetLearner,
                       SingleShotDetectorLearner)):
            self.classes = self.learner.classes
        if isinstance(self.learner, DetrLearner):
            self.classes = [
                "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
                "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A",
                "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush",
            ]

        # Verify that allowed classes are in the detector's class list
        invalid_classes = [cls for cls in self.allowed_classes if cls not in self.classes]
        if invalid_classes:
            raise ValueError(
                f"The following classes are not detected by this detector: {', '.join(invalid_classes)}")

    def infer(self, img=None, threshold=None, keep_size=None, input=None, conf_threshold=None, iou_threshold=None,
              nms_max_num=None, size=None, custom_nms=None, nms_thresh=None, nms_topk=None, post_nms=None,
              extract_maps=None):

        # match variable names
        if isinstance(self.learner, NanodetLearner) and input is not None:
            img = input

        if img is None:
            raise ValueError(
                "An image input is required. Please provide a valid image.")

        if isinstance(self.learner, CenterNetDetectorLearner):
            if threshold is None:
                threshold = 0.2
            if keep_size is None:
                keep_size = True
            boxes = self.learner.infer(img, threshold=threshold, keep_size=keep_size)

        elif isinstance(self.learner, YOLOv3DetectorLearner):
            if threshold is None:
                threshold = 0.1
            if keep_size is None:
                keep_size = True
            boxes = self.learner.infer(img, threshold=threshold, keep_size=keep_size)

        elif isinstance(self.learner, YOLOv5DetectorLearner):
            if size is None:
                size = 640
            boxes = self.learner.infer(img, size)

        elif isinstance(self.learner, DetrLearner):
            boxes = self.learner.infer(img)

        elif isinstance(self.learner, NanodetLearner):
            if conf_threshold is None:
                conf_threshold = 0.35
            if iou_threshold is None:
                iou_threshold = 0.6
            if nms_max_num is None:
                nms_max_num = 100
            boxes = self.learner.infer(img, conf_threshold, iou_threshold, nms_max_num)

        elif isinstance(self.learner, SingleShotDetectorLearner):
            if threshold is None:
                threshold = 0.2
            if keep_size is None:
                keep_size = False
            if nms_thresh is None:
                nms_thresh = 0.45
            if nms_topk is None:
                nms_topk = 400
            if post_nms is None:
                post_nms = 100
            if extract_maps is None:
                extract_maps = False
            boxes = self.learner.infer(img, threshold, keep_size, custom_nms,
                                       nms_thresh, nms_topk, post_nms, extract_maps)
        else:
            raise ValueError(
                "Filtering has not been implemented for the specified detector class."
            )

        if not self.allowed_classes:
            return boxes
        filtered_boxes = BoundingBoxList(
            [box for box in boxes if self.classes[int(box.name)] in self.allowed_classes])
        return filtered_boxes

    def __getattr__(self, attr):
        return getattr(self.learner, attr)
