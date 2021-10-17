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
import random
from shutil import copyfile
import zipfile

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

    def __init__(self, object_name = None, data_directory = None, lr = 0.0008, batch_size = 2,
                    num_workers = 2, num_classes = 1, iters = 1000,
                                            threshold = 0.8,    device = 'cuda'):
        super(SingleDemoGraspLearner, self).__init__(lr = lr, threshold = threshold,
                            batch_size = batch_size, device = device)

        self.object_name = object_name
        self.dataset_dir = os.path.join(data_directory, object_name)
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr_rate = lr
        self.device = device
        self.iters = iters
        self.threshold = threshold
        self.output_dir =  os.path.join(self.dataset_dir, "output")
        self.temp_dir = os.path.join(data_directory, "download_temp")

    def init_model(self):


        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(
                            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

        self.cfg.DATASETS.TRAIN = (self.object_name + "_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = self.num_workers
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        self.cfg.SOLVER.BASE_LR = self.lr_rate
        self.cfg.SOLVER.MAX_ITER = self.iters
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = self.num_kps
        self.cfg.OUTPUT_DIR = self.output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok = True)



    def fit(self):
        self.metadata = self._prepare_datasets(self.object_name)
        self.init_model()
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def infer(self, img_data):

        if isinstance(img_data, Image):
            _img = img.numpy()
        elif isinstance(img, np.ndarray):
            _img = img
        else:
            raise ValueError("Input should be of type Image or numpy array.")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold

        if os.path.exists(os.path.join(self.cfg.OUTPUT_DIR , self.object_name + "/model_final.pth")):

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

        bbx_train = np.load(self.dataset_dir + dataset_name + '/images/annotations/boxes_train.npy', encoding='bytes')
        bbx_val = np.load(self.dataset_dir + dataset_name + '/images/annotations/boxes_val.npy', encoding='bytes')
        kps_train = np.load(self.dataset_dir + dataset_name + '/images/annotations/kps_train.npy', encoding='bytes')
        kps_val = np.load(self.dataset_dir + dataset_name + '/images/annotations/kps_val.npy', encoding='bytes')
        vars()[object_name+'_metadata'], train_set, val_set = register_datasets(DatasetCatalog, MetadataCatalog,
                    self.dataset_dir, dataset_name, bbx_train, kps_train, bbx_val, kps_val)

        self.num_train = len(bbx_train)
        self.num_val = len(bbx_val)
        self.num_kps = len(kps_train[0][0])
        self.train_set = train_set
        self.val_set = val_set
        return vars()[object_name+'_metadata']

    def load(self, path_to_model):

        self.metadata = self._prepare_datasets(self.object_name)
        self.init_model()
        if os.path.isfile(path_to_model):

            print("Found the model, loading...")
            self.cfg.MODEL.WEIGHTS = path_to_model
            self.predictor = DefaultPredictor(self.cfg)

        else:
            assert os.path.isfile(path_to_model), "Checkpoint {} not found!".format(path_to_model)



    def save(self, path):

        if os.path.isfile(os.path.join(self.output_dir, "model_final.pth")):
            print("found the trained model at: " + os.path.join(self.output_dir, "model_final.pth") )
            if path ! = self.output_dir:
                print("copying the trained model to your desired directory at: ")
                print(path)
                shutil.copyfile(os.path.join(self.output_dir, "model_final.pth"), path)
            else:
                print("model is already saved at: " + os.path.join(self.output_dir, "model_final.pth"))
        else:
            print("no trained model was found...")



    def download(self, path=None, verbose=False, object_name=None, url=OPENDR_SERVER_URL +
                                    "/control/single_demo_grasp/"):

        if path is None:
            path = self.temp_dir
        if object_name is None:
            object_name = "pendulum"
        if not os.path.exists(path):
            os.makedirs(path)

        if verbose==True:
            print("Downloading pretrained model, training data and samples for: "+ object_name)

        filename = object_name + ".zip"
        url = os.path.join(url, filename)
        destination_file = os.path.join(path, filename)
        urlretrieve(url, destination_file)

        with zipfile.ZipFile(destination_file, 'r') as zip_ref:
            zip_ref.extractall(path)

        """
        removing zip file after extracting contents
        """
        os.remove(destination_file)


    def eval(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()

    def optimize(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()

    def reset(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()
