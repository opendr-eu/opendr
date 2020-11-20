import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

# OpenDR engine imports
from engine.learners import Learner
from engine.datasets import ExternalDataset, DatasetIterator
from engine.data import Image
from engine.target import Pose


class GluonCVPoseLearner(Learner):
    def __init__(self, lr=4e-5, batch_size=8, lr_schedule='', checkpoint_after_iter=5000, checkpoint_load_iter=0,
                 temp_path="", device="cpu", detector="ssd_512_mobilenet1.0_coco",
                 pose_estimator="simple_pose_resnet18_v1b", pretrained="ccd24037"):
        super(GluonCVPoseLearner, self).__init__(lr=lr, batch_size=batch_size, lr_schedule=lr_schedule,
                                                 checkpoint_after_iter=checkpoint_after_iter,
                                                 checkpoint_load_iter=checkpoint_load_iter,
                                                 temp_path=temp_path, device=device,
                                                 backbone=detector, network_head=pose_estimator)
        if self.device == "cpu":
            self.ctx = mx.cpu()
        elif self.device == "cuda":
            self.ctx = mx.gpu()

        # Detector setup
        self.backbone = detector.lower()
        # TODO add more from https://cv.gluon.ai/model_zoo/detection.html
        supportedBackbones = ["ssd_512_mobilenet1.0_coco", "yolo3_darknet53_coco"]
        if self.backbone not in supportedBackbones:
            raise ValueError(self.backbone + " not a valid detector. Supported detectors:" + str(supportedBackbones))
        if "ssd" in self.backbone:
            self.transform = gcv.data.transforms.presets.ssd.transform_test
        elif "yolo" in self.backbone:
            self.transform = gcv.data.transforms.presets.yolo.transform_test

        # Pose estimator setup
        self.network_head = pose_estimator.lower()
        # TODO add more from https://cv.gluon.ai/model_zoo/pose.html#id46
        supportedEstimators = ["simple_pose_resnet18_v1b", "mobile_pose_mobilenetv3_small"]
        if self.network_head not in supportedEstimators:
            raise ValueError(self.network_head + " not a valid estimator. Supported estimators:" +
                             str(supportedEstimators))
        # Pose estimator dimensions setup
        self.dims = [256, 192]
        # Outlier dimensions based on model hashtags
        models_dimensions_index = {"ccd24037": [128, 96], "2f544338": [384, 288]}
        if type(pretrained) == bool:
            self.pretrained = pretrained
        else:
            self.pretrained = pretrained
            # Handle outlier dimensions based on
            if self.pretrained in models_dimensions_index.keys():
                self.dims = models_dimensions_index[self.pretrained]

        self.model = get_model(self.network_head, pretrained=self.pretrained, ctx=self.ctx)
        self.model.hybridize()

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        pass

    def eval(self, dataset):
        pass

    def infer(self, batch):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def optimize(self, target_device):
        pass

    def reset(self):
        pass
