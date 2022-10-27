# Copyright 2020-2022 OpenDR European Project
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
import numpy as np
import zipfile
from urllib.request import urlretrieve
import shutil

# Detectron imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.data import Image

# single demo grasp module imports
from opendr.control.single_demo_grasp.training.learner_utils import register_datasets


class SingleDemoGraspLearner(Learner):
    def __init__(self, object_name=None, data_directory=None, lr=0.0008, batch_size=512, img_per_step=2, num_workers=2,
                 num_classes=1, iters=1000, threshold=0.8, device='cuda'):
        super(SingleDemoGraspLearner, self).__init__(lr=lr, threshold=threshold, batch_size=batch_size, device=device,
                                                     iters=iters)
        self.dataset_dir = data_directory
        self.object_name = object_name
        self.output_dir = os.path.join(self.dataset_dir, self.object_name, "output")
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.temp_dir = os.path.join(self.dataset_dir, "download_temp")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATALOADER.NUM_WORKERS = self.num_workers
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = device
        self.cfg.SOLVER.IMS_PER_BATCH = img_per_step
        self.cfg.SOLVER.BASE_LR = lr
        self.cfg.SOLVER.MAX_ITER = iters
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
        self.cfg.OUTPUT_DIR = self.output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def fit(self):
        self.metadata = self._prepare_datasets()
        self.cfg.DATASETS.TRAIN = (self.object_name + "_train",)
        self.cfg.DATASETS.TEST = ()

        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()

    def infer(self, img_data):
        if not isinstance(img_data, Image):
            img_data = Image(img_data)
        img_data = img_data.convert(format='channels_last', channel_order='rgb')

        self.predictor = DefaultPredictor(self.cfg)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold

        output = self.predictor(img_data)
        bounding_box = output["instances"].to("cpu").pred_boxes.tensor.numpy()
        keypoints_pred = output["instances"].to("cpu").pred_keypoints.numpy()

        if len(bounding_box) > 0:
            return 1, bounding_box[0], keypoints_pred[0]
        else:
            return 0, None, None

    def _prepare_datasets(self):
        bbx_train = np.load(os.path.join(self.dataset_dir, self.object_name, 'images/annotations/boxes_train.npy'),
                            encoding='bytes')
        bbx_val = np.load(os.path.join(self.dataset_dir, self.object_name, 'images/annotations/boxes_val.npy'),
                          encoding='bytes')
        kps_train = np.load(os.path.join(self.dataset_dir, self.object_name, 'images/annotations/kps_train.npy'),
                            encoding='bytes')
        kps_val = np.load(os.path.join(self.dataset_dir, self.object_name, 'images/annotations/kps_val.npy'),
                          encoding='bytes')
        vars()[self.object_name + '_metadata'], train_set, val_set = register_datasets(DatasetCatalog, MetadataCatalog,
                                                                                       self.dataset_dir, self.object_name,
                                                                                       bbx_train, kps_train, bbx_val, kps_val)

        self.num_train = len(bbx_train)
        self.num_val = len(bbx_val)
        self.num_kps = len(kps_train[0][0])
        self.train_set = train_set
        self.val_set = val_set
        return vars()[self.object_name + '_metadata']

    def load(self, path_to_model):
        if os.path.isfile(path_to_model):
            self.cfg.MODEL.WEIGHTS = path_to_model
            self.predictor = DefaultPredictor(self.cfg)
            print("Model loaded!")
        else:
            assert os.path.isfile(path_to_model), "Checkpoint {} not found!".format(path_to_model)

    def save(self, path):
        if os.path.isfile(os.path.join(self.output_dir, "model_final.pth")):
            print("found the trained model at: " + os.path.join(self.output_dir, "model_final.pth"))
            if path != self.output_dir:
                print("copying the trained model to your desired directory at: ")
                print(path)
                shutil.copyfile(os.path.join(self.output_dir, "model_final.pth"), os.path.join(path, "model_final.pth"))
            else:
                print("model is already saved at: " + os.path.join(self.output_dir, "model_final.pth"))
        else:
            print("no trained model was found...")

    def download(self, path=None, object_name=None):
        if path is None:
            path = self.temp_dir
        if object_name is None:
            object_name = "pendulum"
        if not os.path.exists(path):
            os.makedirs(path)

        print("Downloading pretrained model, training data and samples for: " + object_name)

        filename = object_name + ".zip"
        url = os.path.join(OPENDR_SERVER_URL, "control/single_demo_grasp/", filename)
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
