#!/usr/bin/env python

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

import time
import os
import sys
import numpy as np
import detectron2
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg

from inference_utils import *
from opendr.control.single_demo_grasp.training.single_demo_grasp_learner import SingleDemoGraspLearner
from opendr.engine.data import Image

# variable definitions here
dir_temp = os.path.join("./", "sdg_temp")


class SingleDemoInference(object):

    def __init__(self, model_filepath, thresh):
    
        self.learner = SingleDemoGraspLearner(object_name = 'pendulum', data_directory = dir_temp, device='cpu')
        self.learner.download(path = dir_temp, object_name = "pendulum")
        self.learner.load(model_filepath)

    def predict(self, data):
        flag, bounding_box, keypoints_pred =self.learner.infer(data)

        if flag == 1:
            return 1, np.array(bounding_box), correct_orientation_ref(
                get_angle(keypoints_pred,1)), get_kps_center(keypoints_pred)
        else:
            print("No detection")
            return 0, [-1,-1,-1,-1] , -1, [-1, -1]

if __name__ == '__main__':

    myimage = Image.open("samples/0.jpg")
    clone = myimage.opencv()
    model = SingleDemoInference(model_filepath = os.path.join(dir_temp, "pendulum", "output", "model_final.pth"), thresh=0.8)
    _, outputs, out_kps, kps_ctr = model.predict(myimage)


    center_coordinates = (kps_ctr[0], kps_ctr[1])
    radius = 10
    color = (255, 0, 0)
    thickness = 2
    clone = cv2.circle(clone, center_coordinates, radius, color, thickness)
    out_img = cv2.rectangle(clone, (outputs[0], outputs[1]), (outputs[2],
                            outputs[3]), (255,0,0), 2)
    cv2.imshow("print bbox", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
