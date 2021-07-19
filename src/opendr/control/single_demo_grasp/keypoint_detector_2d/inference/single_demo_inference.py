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
import cv2
import os
import sys
import numpy as np
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from inference_utils import *


class SingleDemoInference(object):

    def __init__(self, model_filepath, thresh):
        self.model_filepath = model_filepath
        self.thresh = thresh
        self.load_graph()

    def load_graph(self):
        model_zoo.get_config_file
        self.cfg=get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(
                            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.thresh
        self.cfg.SOLVER.BASE_LR = 0.0008
        self.cfg.SOLVER.MAX_ITER = 1000
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
        self.cfg.MODEL.DEVICE="cuda"
        self.cfg.INPUT.MIN_SIZE_TEST=0
        self.cfg.MODEL.WEIGHTS = os.path.join("",self.model_filepath)
        self.predictor = DefaultPredictor(self.cfg)
        print('Model loading complete!')

    def predict(self, data):
        output=self.predictor(data)
        print(output)
        bounding_box = output["instances"].to("cpu").pred_boxes.tensor.numpy()
        keypoints_pred =  output["instances"].to("cpu").pred_keypoints.numpy()

        if len(bounding_box)>0:
            return 1, np.array(bounding_box[0]), correct_orientation_ref(
                get_angle(keypoints_pred,1)), get_kps_center(keypoints_pred)
        else:
            print("No detection")
            return 0, [-1,-1,-1,-1] , -1, [-1, -1]

if __name__ == '__main__':

    myimage = cv2.imread("samples/0.jpg")
    clone = myimage.copy()
    model = SingleDemoInference("models/pendulum_model.pth")
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
