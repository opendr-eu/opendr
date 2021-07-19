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



import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Detectron imports
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.constants import OPENDR_SERVER_URL

# single demo grasp modul imports
from opendr.control.single_demo_grasp.keypoint_detector_2d.training.learner_utils import *


class SingleDemoGraspLearner(Learner):

    def __init__(self, object_name = '', dataset_dir = '', lr = 0.0008, batch_size = 4,
                    num_workers = 2, num_classes = 1, iters = 1000,
                                            threshold = 0.8,    device = 'cuda'):
        super(SingleDemoGraspLearner, self).__init__(lr = lr, threshold = threshold
                            batch_size = batch_size, device = device)

        self.object_name = object_name
        self.image_dir = dataset_dir
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr_rate = lr
        self.device = device
        self.iters = iters
        self.threshold = threshold

    def init_model(self, img_per_batch, lr_rate, max_iterations):


        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(
                            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

        self.cfg.DATASETS.TRAIN = (self.object_name + "_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = self.num_workers
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.SOLVER.IMS_PER_BATCH = img_per_batch
        self.cfg.SOLVER.BASE_LR = lr_rate
        self.cfg.SOLVER.MAX_ITER = iters
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = self.num_kps
        self.cfg.OUTPUT_DIR = os.getcwd() + "/output/" + self.object_name + "/"
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok = True)



    def fit(self):

        self.init_model()
        metadata = _prepare_dataset(self.object_name)
        trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()

    def infer(self, img_data):

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold

        if os.path.exists(os.path.join(os.getcwd() + "/output/" + self.object_name, "model_final.pth")):

            print("Found the model, loading...")
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
            self.predictor = DefaultPredictor(self.cfg)

        else:
            print ("No trained model was found!")
            return -1

        output = self.predictor(img_data)
        bounding_box = output["instances"].to("cpu").pred_boxes.tensor.numpy()
        keypoints_pred =  output["instances"].to("cpu").pred_keypoints.numpy()

        if len(bounding_box)>0:
            return 1, bounding_box[0], keypoints_pred[0]
        else:
            return 0, -1, -1

    def infer_raw_output(self, img_data):

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold

        if os.path.exists(os.path.join(os.getcwd() + "/output/" + self.object_name, "model_final.pth")):

            print("Found the model, loading...")
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
            self.predictor = DefaultPredictor(self.cfg)

        else:
            print ("No trained model was found!")
            return -1

        output = self.predictor(img_data)
        bounding_box = output["instances"].to("cpu").pred_boxes.tensor.numpy()

        if len(bounding_box)>0:
            return 1, output
        else:
            return 0, -1


    def _prepare_datasets(self, dataset_name):

        bbx_train = np.load(self.image_dir + dataset_name + '/annotations/boxes_train.npy', encoding='bytes')
        bbx_val = np.load(self.image_dir + dataset_name + '/annotations/boxes_val.npy', encoding='bytes')
        kps_train = np.load(self.image_dir + dataset_name + '/annotations/kps_train.npy', encoding='bytes')
        kps_val = np.load(self.image_dir + dataset_name + '/annotations/kps_val.npy', encoding='bytes')
        vars()[object_name+'_metadata'] = register_datasets(DatasetCatalog, MetadataCatalog,
                    self.image_dir, dataset_name, bbx_train, kps_train, bbx_val, kps_val)

        self.num_train = len(self.bbx_train)
        self.num_val = len(self.bbx_val)
        self.num_kps = len(self.kps_train[0][0])

        return vars()[object_name+'_metadata']

    def load(self, path_to_model):

        if os.path.isfile(path_to_model):

            print("Found the model, loading...")
            self.cfg.MODEL.WEIGHTS = path_to_model
            self.predictor = DefaultPredictor(self.cfg)

        else:
            assert os.path.isfile(path_to_model), "Checkpoint {} not found!".format(path_to_model)



    def save(self):

        raise NotImplementedError()

    def optimize(self):

        raise NotImplementedError()

    def reset(self):

        raise NotImplementedError()

    def download(self):
        # TODO
        assert NotImplementedError()

    def eval(self):

        raise NotImplementedError()
