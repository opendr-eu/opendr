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

import cv2
import numpy as np


def visualize(learner, image_path):
    """
    Visualizes the output of a fitted high resolution Learner on a given input image using OpenCV
    :param learner: the fitted Learner
    :param image_path: path of image to
    :return:
    """

    img = cv2.imread(image_path)
    heatmap = learner.infer(img)
    heatmap = cv2.normalize(heatmap.data, None, 0, 1, cv2.NORM_MINMAX)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    cv2.imshow("Output Image", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
