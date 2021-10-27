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

"""
Misc functions, for drawing results.
"""

import cv2


def plot_results(image, boxes, w_sensor1, fps=None, classes=None, colors=None):
    """
    Helper function for creating the annotated images.
    :param image: Image that is to be annotated
    :type image: numpy.ndarray
    :param boxes: List of detected bounding boxes
    :type boxes: opendr.engine.target.BoundingBoxList
    :param w_sensor1: Weight of the first modality (color image)
    :type w_sensor1: float
    :param fps: Frames per second
    :type fps: int, defaults to None
    :param classes: Classes of objects, defaults to None
    :type classes: list
    :param colors: Colors for visualization, defaults to None
    :type colors: list
    :return: Image with annotations
    :rtype: numpy.ndarray
    """
    # l515_dataset classes
    if classes is None:
        classes = ['chair', 'cycle', 'bin', 'laptop', 'drill', 'rocker', 'can']

    # colors for visualization
    if colors is None:
        colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    if fps is not None:
        cv2.putText(image, "FPS: {:.1f}".format(fps), (1100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    for box, c in zip(boxes, colors * 100):
        if box.confidence > 0.7:
            top_left = [int(box.left), int(box.top)]
            bottom_right = [int(box.left + box.width), int(box.top + box.height)]
            cv2.rectangle(image, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (51, 102, 255), 2)
            cv2.rectangle(image, (0, 0), (int(image.shape[1] * w_sensor1), 30), (255, 0, 0), -1)
            cv2.rectangle(image, (int(image.shape[1] * w_sensor1), 0), (image.shape[1], 30), (51, 153, 51), -1)

            text = f'{classes[box.name - 1]}: {box.confidence:0.2f}'
            cv2.putText(image, text, (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image
