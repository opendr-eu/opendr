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

from logging import warning
from os.path import join

import numpy as np
from ultralytics import YOLO

# OpenDR engine imports
from opendr.engine.data import Image
from opendr.engine.learners import Learner
from opendr.engine.target import BoundingBox, BoundingBoxList, Heatmap


class YOLOv8SegLearner(Learner):
    available_models = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg", "custom"]

    def __init__(self, model_name, model_path=None, device='cuda', temp_path='.'):
        super(YOLOv8SegLearner, self).__init__(device=device, temp_path=temp_path)
        if model_name not in self.available_models:
            model_name = 'yolov8s-seg'
            print('Unrecognized model name, defaulting to "yolov8s-seg"')

        if model_name == "custom" and model_path is None:
            raise ValueError("When 'model_name' is set to 'custom', a valid model_path to a custom model is required.")

        if model_path is None:
            self.model = YOLO(join(self.temp_path, model_name + ".pt"))  # load an official model
        else:
            self.model = YOLO(model_path)  # load a custom model

        self.model.to(device)
        self.classes = [self.model.names[i] for i in range(len(self.model.names.keys()))]

        self.results = None

    def infer(self, img, conf_thres=0.25, iou_thres=0.7, image_size=None, half_prec=False,
              agnostic_nms=False, classes=None, no_mismatch=False, verbose=False, show=False):
        """
        Runs inference using the loaded model.

        :param img: The image to run inference on
        :type img: opendr.engine.data.Image
        :param conf_thres: Object confidence threshold for detection, defaults to '0.25'
        :type conf_thres: float, optional
        :param iou_thres: Intersection over union (IoU) threshold for NMS, defaults to '0.7'
        :type iou_thres: float, optional
        :param image_size: Image size as scalar or (h, w) list, i.e. (640, 480), defaults to 'None'
        :type image_size: int or tuple, optional
        :param half_prec: Use half precision (FP16), defaults to 'False'
        :type half_prec: bool, optional
        :param agnostic_nms: Class-agnostic NMS, defaults to 'False'
        :type agnostic_nms: bool, optional
        :param classes: Filter results by class, i.e. classes=["person", "chair"], defaults to 'None'
        :type classes: list, optional
        :param no_mismatch: Whether to check and warn for mismatch between input image
                        size and output heatmap size, defaults to 'False'
        :type no_mismatch: bool, optional
        :param verbose: Whether to print YOLOv8 prediction information, defaults to 'False'
        :type verbose: bool, optional
        :param show: Whether to use the YOLOv8 built-in visualization feature of predict, defaults to 'False'
        :type show: bool, optional

        :return: The detected semantic segmentation OpenDR heatmap
        :rtype: opendr.engine.target.Heatmap
        """
        if image_size is None:
            image_size = (480, 640)

        if not isinstance(img, Image) and not isinstance(img, str):
            img = Image(img)  # NOQA

        # Class filtering
        class_inds = None
        if classes is not None:
            # list out keys and values separately
            key_list = list(self.model.names.keys())
            val_list = list(self.model.names.values())
            class_inds = []
            for class_name in classes:
                try:
                    class_inds.append(key_list[val_list.index(class_name)])
                except ValueError:
                    raise ValueError(f"\"{class_name}\" is not in detectable classes. Check out the available classes with "
                                     f"get_classes().")

        # https://docs.ultralytics.com/modes/predict/#inference-arguments
        if isinstance(img, Image):
            self.results = self.model.predict(img.opencv(), save=False, verbose=verbose, device=self.device,
                                              imgsz=image_size, conf=conf_thres, iou=iou_thres, half=half_prec,
                                              agnostic_nms=agnostic_nms, classes=class_inds, show=show)
        elif isinstance(img, str):
            # Take advantage of YOLOv8 built-in features, see https://docs.ultralytics.com/modes/predict/#inference-sources
            self.results = self.model.predict(img, save=False, verbose=verbose, device=self.device,
                                              imgsz=image_size, conf=conf_thres, iou=iou_thres, half=half_prec,
                                              agnostic_nms=agnostic_nms, classes=class_inds, show=show)
        heatmap = self.__get_opendr_heatmap(no_mismatch)

        return heatmap

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        """This method is not used in this implementation."""
        raise NotImplementedError

    def eval(self, dataset):
        """This method is not used in this implementation."""
        raise NotImplementedError

    def optimize(self, target_device):
        """This method is not used in this implementation."""
        return NotImplementedError

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def save(self, path):
        """This method is not used in this implementation."""
        return NotImplementedError

    def load(self, path):
        """This method is not used in this implementation."""
        return NotImplementedError

    def get_bboxes(self):
        """
        This method converts the latest infer results to OpenDR bounding boxes and returns them.
        This serves as a helpful method to perform object detection along the semantic segmentation.

        :return: The Opendr bounding box list
        :rtype: opendr.engine.target.BoundingBoxList
        """
        if self.results is None:
            raise Warning("\"self.results\" is None, please run infer() first.")

        bounding_boxes = BoundingBoxList([])
        for yolo_box in self.results[0].boxes:
            box_cpu = yolo_box.xywh[0].cpu().numpy()
            box_class = yolo_box.cls.cpu().numpy()[0]
            box_conf = yolo_box.conf.cpu().numpy()[0]
            bbox = BoundingBox(left=box_cpu[0], top=box_cpu[1],
                               width=box_cpu[2], height=box_cpu[3],
                               name=box_class, score=box_conf)
            bounding_boxes.data.append(bbox)

        return bounding_boxes

    def get_visualization(self, labels=True, boxes=True, masks=True, conf=True):
        """
        Returns the annotated frame from the latest detection.

        :param labels:  Whether to plot the label of bounding boxes, defaults to 'True'
        :type labels: bool, optional
        :param boxes: Whether to plot the bounding boxes, defaults to 'True'
        :type boxes: bool, optional
        :param masks: Whether to plot the masks, defaults to 'True'
        :type masks: bool, optional
        :param conf: Whether to plot the detection confidence score
        :type conf: bool, optional

        :return: A numpy array of the annotated image.
        :rtype: numpy.ndarray
        """
        return self.results[0].plot(labels=labels, boxes=boxes, masks=masks, conf=conf)

    def get_classes(self):
        """
        Returns the available class names of the loaded model.

        :return: Dictionary of the class names
        :rtype: dict
        """
        return self.model.names

    def __get_opendr_heatmap(self, no_mismatch):
        """
        Converts the YOLOv8 output (ultralytics.engine.results.Results) to opendr.engine.target.Heatmap and returns it.

        :param no_mismatch: Whether to  check for heatmap.shape and image orig_shape mismatch
        :type no_mismatch: bool
        :return: The OpenDR Heatmap
        :rtype: opendr.engine.target.Heatmap
        """
        try:
            # Attempt to initialize heatmap shape from detected mask to match image_size requested by user
            heatmap_img = np.zeros(self.results[0].masks[0].data[0].shape)
        except TypeError:
            # Fallback to original image shape if no mask is detected, which avoid shape mismatch error
            heatmap_img = np.zeros(self.results[0].orig_shape)

        if self.results[0].masks is not None:  # Failsafe for no detections
            for i in range(len(self.results[0].masks)):
                mask = self.results[0].masks[i].data[0].cpu().numpy()
                mask_class = int(self.results[0].boxes[i].cls.cpu().numpy()[0])  # Class name taken from box
                class_names = self.results[0].names.copy()

                # Person class index is 0, need to replace
                if mask_class == 0:
                    mask_class = list(self.results[0].names.keys())[-1] + 1
                    # Need to modify the class name dictionary too
                    class_names[mask_class] = 'person'
                    class_names[0] = 'background'

                mask[mask == 1] = mask_class  # Replace 1s with actual class indices
                np.putmask(heatmap_img, mask.astype(bool), mask)  # Add this detection's mask on the heatmap

        if heatmap_img.shape != self.results[0].orig_shape and not no_mismatch:
            warning(f"Mismatch between original image shape {self.results[0].orig_shape} and output heatmap shape "
                    f"{heatmap_img.shape}, please modify your image_size argument or pass no_mismatch=True "
                    f"to silence this warning.")

        return Heatmap(data=heatmap_img.astype(int), class_names=self.results[0].names)
