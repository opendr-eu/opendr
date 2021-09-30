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

import matplotlib.pyplot as plt
import numpy as np
import cv2

from opendr.engine.data import Image
from opendr.engine.target import BoundingBoxList
from opendr.perception.object_detection_2d.datasets.transforms import BoundingBoxListToNumpyArray

np.random.seed(0)


def get_unique_color(index, num_colors, cmap='jet'):
    """
    Generates num_colors of unique colors from the given cmap and returns the color at index.
    """
    colors = plt.get_cmap(cmap)
    c = [int(x * 255) for x in colors(index / float(num_colors))[:3]][::-1]
    return c


VOC_COLORS = [get_unique_color(i, 20) for i in range(20)]
np.random.shuffle(VOC_COLORS)


def get_dataset_palette(n_classes):
    """
    Generates a palette for n_classes.
    """
    palette = [get_unique_color(i, n_classes) for i in range(n_classes)]
    return palette


def draw_bounding_boxes(img, bounding_boxes, class_names=None, show=False, line_thickness=None):
    """
    :param img: image on which to draw bounding boxes
    :type img: opendr.engine.data.Image
    :param bounding_boxes: detected or groundtruth bounding boxes
    :type bounding_boxes: opendr.engine.target.BoundingBoxList
    :param class_names: list of class names to be drawn on top of the bounding boxes
    :type class_names: list
    :param show: whether to display the resulting annotated image or not
    :type show: bool
    :param line_thickness: line thickness in pixels
    :type line_thickness: int
    """
    if isinstance(img, Image):
        img = img.data
    assert isinstance(bounding_boxes, BoundingBoxList), "bounding_boxes must be of BoundingBoxList type"

    if not bounding_boxes.data:
        return draw_detections(img, np.empty((0, 4)), np.empty((0,)), np.empty((0,)), class_names, show, line_thickness)

    bounding_boxes = BoundingBoxListToNumpyArray()(bounding_boxes)
    boxes = bounding_boxes[:, :4]
    scores = bounding_boxes[:, 4]
    classes = bounding_boxes[:, 5].astype(np.int)
    return draw_detections(img, boxes, scores, classes, class_names, show, line_thickness)


def draw_detections(img, boxes, scores, classes, class_names=None, show=False, line_thickness=None):
    """
    :param img: image on which to draw bounding boxes
    :type img: np.ndarray or opendr.engine.data.Image
    :param boxes: bounding boxes in numpy array or list format [n, 4] (coordinate format: x1, y1, x2, y2)
    :type boxes: np.ndarray or list
    :param scores: confidence scores for each bounding box [n]
    :type scores: np.ndarray or list
    :param classes: class indices for each bounding box [n]
    :type classes: np.ndarray or list
    :param show: whether to display the resulting annotated image or not
    :type show: bool
    :param line_thickness: line thickness in pixels
    :type line_thickness: int
    """
    if isinstance(img, Image):
        img = img.data
    # boxes in x1, y1, x2, y2 list format [n, 4]
    # scores and classes in [n] list format
    classes = np.int32(classes)
    palette = VOC_COLORS
    n_classes = len(palette)

    for idx, pred_box in enumerate(boxes):
        # pred_box_w, pred_box_h = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]
        tl = line_thickness or int(0.003 * max(img.shape[:2]))
        c1 = (max(0, int(pred_box[0])), max(0, int(pred_box[1])))
        c2 = (min(img.shape[1], int(pred_box[2])), min(img.shape[0], int(pred_box[3])))
        color = tuple(palette[classes[idx] % n_classes])

        img = np.ascontiguousarray(img, dtype=np.uint8)
        cv2.rectangle(img, c1, c2, color, thickness=2)

        if class_names is not None:
            label = "{}".format(class_names[classes[idx]])

            t_size = cv2.getTextSize(
                label, 0, fontScale=float(tl) / 5, thickness=1)[0]

            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            t = - 2
            if c2[1] < 0:
                c2 = c1[0] + t_size[0], c1[1] + t_size[1]
                t = t_size[1] - 4
            cv2.rectangle(img, c1, c2, color, -1)  # filled

            cv2.putText(img, label, (c1[0], c1[1] + t), 0, float(tl) / 5,
                        [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    if show:
        cv2.imshow('detections', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img
