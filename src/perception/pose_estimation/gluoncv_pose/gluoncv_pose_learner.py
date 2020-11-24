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

        self.detector = get_model(self.backbone, pretrained=True, ctx=self.ctx)
        self.detector.reset_class(classes=['person'], reuse_weights={'person': 'person'})  # Keep only person class
        self.detector.hybridize()

        # Pose estimator setup
        self.network_head = pose_estimator.lower()
        # TODO add more from https://cv.gluon.ai/model_zoo/pose.html#id46
        supportedEstimators = ["simple_pose_resnet18_v1b", "mobile_pose_mobilenetv3_small"]
        if self.network_head not in supportedEstimators:
            raise ValueError(self.network_head + " not a valid estimator. Supported estimators:" +
                             str(supportedEstimators))
        # Pose estimator dimensions setup
        self.dims = (256, 192)
        # Outlier dimensions based on model hashtags
        models_dimensions_index = {"ccd24037": (128, 96), "2f544338": (384, 288)}
        if type(pretrained) == bool:
            self.pretrained = pretrained
        else:
            self.pretrained = pretrained
            # Handle outlier dimensions based on hashtag
            if self.pretrained in models_dimensions_index.keys():
                self.dims = models_dimensions_index[self.pretrained]

        self.model = get_model(self.network_head, pretrained=self.pretrained, ctx=self.ctx)
        self.model.hybridize()

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        pass

    def eval(self, dataset):
        pass

    def infer(self, img, return_extra=False):
        if not isinstance(img, Image):
            img = Image(img)
        img = img.numpy()

        img = mx.nd.array(img).astype('uint8')

        x, img = self.transform(img, short=512, max_size=350)
        x = x.as_in_context(self.ctx)

        class_IDs, scores, bounding_boxes = self.detector(x)

        pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxes,
                                                           output_shape=self.dims, ctx=self.ctx)

        poses = []
        if len(upscale_bbox) > 0:
            predicted_heatmap = self.model(pose_input)
            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
            for pose_coords, pose_conf in zip(pred_coords, confidence):
                pose = Pose(pose_coords.asnumpy(), pose_conf.asnumpy())
                # for keypoint, conf in zip(pose_coords, pose_conf):
                #     print(keypoint.asnumpy(), conf.asnumpy())
                    # 0, nose
                    # 1, l_eye
                    # 2, r_eye
                    # 3, l_ear
                    # 4, r_ear
                    # 5, l_sho
                    # 6, r_sho
                    # 7, l_elb
                    # 8, r_elb
                    # 9, l_wri
                    # 10, r_wri
                    # 11, l_hip
                    # 12, r_hip
                    # 13, l_knee
                    # 14, r_knee
                    # 15, l_ank
                    # 16, r_ank
                poses.append(pose)
        if return_extra:
            return poses, class_IDs, bounding_boxes, scores, img
        else:
            return poses

    def save(self, path):
        pass

    def load(self, path):
        pass

    def optimize(self, target_device):
        pass

    def reset(self):
        pass


# learner = GluonCVPoseLearner()
# print(learner.dims)

