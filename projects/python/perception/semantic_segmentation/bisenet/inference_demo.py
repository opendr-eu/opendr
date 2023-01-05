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
from opendr.perception.semantic_segmentation import BisenetLearner
from opendr.engine.data import Image
from matplotlib import cm
import numpy as np

if __name__ == '__main__':
    learner = BisenetLearner()

    # Dowload the pretrained model
    learner.download('./bisenet_camvid', mode='pretrained')
    learner.load('./bisenet_camvid')

    # Download testing image
    learner.download('./', mode='testingImage')
    img = Image.open("./test1.png")

    # Perform inference
    heatmap = learner.infer(img)

    # Create a color map and translate colors
    segmentation_mask = heatmap.data

    colormap = cm.get_cmap('viridis', 12).colors
    segmentation_img = np.uint8(255*colormap[segmentation_mask][:, :, :3])

    # Blend original image and the segmentation mask
    blended_img = np.uint8(0.4*img.opencv() + 0.6*segmentation_img)

    cv2.imshow('Heatmap', blended_img)
    cv2.waitKey(-1)
