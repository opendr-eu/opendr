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

from os.path import join
from logging import warning

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
              agnostic_nms=False, retina_masks=False, classes=None, no_mismatch=False):
        """
        TODO docstring

        source 	str 	'ultralytics/assets' 	source directory for images or videos
        conf 	float 	0.25 	object confidence threshold for detection
        iou 	float 	0.7 	intersection over union (IoU) threshold for NMS
        imgsz 	int or tuple 	640 	image size as scalar or (h, w) list, i.e. (640, 480)
        half 	bool 	False 	use half precision (FP16)
        device 	None or str 	None 	device to run on, i.e. cuda device=0/1/2/3 or device=cpu
        show 	bool 	False 	show results if possible
        save 	bool 	False 	save images with results
        save_txt 	bool 	False 	save results as .txt file
        save_conf 	bool 	False 	save results with confidence scores
        save_crop 	bool 	False 	save cropped images with results
        hide_labels 	bool 	False 	hide labels
        hide_conf 	bool 	False 	hide confidence scores
        max_det 	int 	300 	maximum number of detections per image
        vid_stride 	bool 	False 	video frame-rate stride
        stream_buffer 	bool 	False 	buffer all streaming frames (True) or return the most recent frame (False)
        line_width 	None or int 	None 	The line width of the bounding boxes. If None, it is scaled to the image size.
        visualize 	bool 	False 	visualize model features
        augment 	bool 	False 	apply image augmentation to prediction sources
        agnostic_nms 	bool 	False 	class-agnostic NMS
        retina_masks 	bool 	False 	use high-resolution segmentation masks
        classes 	None or list 	None 	filter results by class, i.e. classes=0, or classes=[0,2,3]
        boxes 	bool 	True 	Show boxes in segmentation predictions

        :param img:
        :return:
        """
        if image_size is None:
            image_size = (480, 640)

        if not isinstance(img, Image):
            img = Image(img)

        # TODO Check for classes filtering
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
        self.results = self.model.predict(img.opencv(), save=False, verbose=False, device=self.device,
                                          imgsz=image_size, conf=conf_thres, iou=iou_thres, half=half_prec,
                                          agnostic_nms=agnostic_nms, retina_masks=retina_masks, classes=class_inds)
        heatmap = self.__get_opendr_heatmap(no_mismatch)

        return heatmap

    def fit(self):
        """This method is not used in this implementation."""
        # TODO investigate if we can wrap training
        raise NotImplementedError

    def eval(self):
        """This method is not used in this implementation."""
        # TODO investigate if we can wrap eval
        raise NotImplementedError

    def optimize(self, target_device):
        """This method is not used in this implementation."""
        # TODO investigate if we can wrap export to optimized models
        return NotImplementedError

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def load(self):
        """This method is not used in this implementation."""
        # TODO should this be used?
        return NotImplementedError

    def save(self):
        """This method is not used in this implementation."""
        # TODO should this be used?
        return NotImplementedError

    def get_bboxes(self):
        """
        TODO
        :return:
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
        :param conf: Whether to plot the detection confidence score.
        :type conf: bool, optional

        :return: A numpy array of the annotated image.
        :rtype: numpy.ndarray
        """
        return self.results[0].plot(labels=labels, boxes=boxes, masks=masks, conf=conf)

    def get_classes(self):
        """
        Returns the available class names of the loaded model.

        :return: Dictionary of the class names.
        :rtype: dict
        """
        return self.model.names

    def __get_opendr_heatmap(self, no_mismatch):
        """
        Converts the YOLOv8 output (ultralytics.engine.results.Results) to opendr.engine.target.Heatmap and returns it.

        :param no_mismatch: Whether to  check for heatmap.shape and image orig_shape mismatch.
        :type no_mismatch: bool
        :return: The OpenDR Heatmap.
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
                    # TODO Add note to docs

                mask[mask == 1] = mask_class  # Replace 1s with actual class indices
                np.putmask(heatmap_img, mask.astype(bool), mask)  # Add this detection's mask on the heatmap

        if heatmap_img.shape != self.results[0].orig_shape and not no_mismatch:
            warning(f"Mismatch between original image shape {self.results[0].orig_shape} and output heatmap shape "
                    f"{heatmap_img.shape}, please modify your image_size argument or pass no_mismatch=True "
                    f"to silence this warning.")

        return Heatmap(data=heatmap_img.astype(int), class_names=self.results[0].names)
