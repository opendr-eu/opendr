# Copyright 2020-2021 OpenDR European Project

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

import cv2
from opendr.engine.target import CocoBoundingBox
import numpy as np
from copy import deepcopy


def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=3,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0)
):
    # Copied from https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size


def draw(img, boxes, make_copy=False, classes=None):
    if make_copy:
        img = deepcopy(img)
    # colors for visualization
    if classes is None:
        classes = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']
    colors = [(0, 44, 74), (85, 32, 9), (92, 69, 12),
              (49, 18, 55), (46, 67, 18), (30, 74, 93)]
    text_list = []
    position_list = []
    for box, c in zip(boxes, colors):
        if box.name >= len(classes):
            continue
        start_point = (int(box.left), int(box.top))
        position_list.append(start_point)
        w, h = int(box.width), int(box.height)
        end_point = (int(box.left) + w, int(box.top) + h)
        text_list.append(f'{classes[box.name]}: {box.confidence:0.2f}')
        if type(box) == CocoBoundingBox and len(box.segmentation) > 4:
            poly = np.array(box.segmentation, dtype=np.int32).reshape((int(len(box.segmentation) / 2), 2))
            blk = np.zeros(img.shape, np.uint8)
            cv2.polylines(img, [poly], True, c, 5)
            cv2.fillPoly(blk, [poly], c)
            img = cv2.addWeighted(img, 1.0, blk, 2.0, 1)
        else:
            cv2.rectangle(img, start_point, end_point, c, 3)
    for idx, text in enumerate(text_list):
        pos = position_list[idx]
        draw_text(img, text, pos=pos, font_thickness=3, font_scale=2)
    return img
