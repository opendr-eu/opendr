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

# fiftyone imports 
# import fiftyone as fo
import fiftyone.utils.random as four

# Detectron imports

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# OpenDR engine imports
from opendr.engine.data import Image
from opendr.engine.learners import Learner
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.target import BoundingBox


class Detectron2Learner(Learner):

    def __init__(self, lr=0.00025, batch_size=200, img_per_step=2, weight_decay=0.00008,
                 momentum=0.98, gamma=0.0005, norm="GN", num_workers=2, num_keypoints=25, 
                 iters=4000, threshold=0.8, loss_weight=1.0, device='cuda', temp_path="temp"):
        super(Detectron2Learner, self).__init__(lr=lr, threshold=threshold, 
                                                batch_size=batch_size, device=device, 
                                                iters=iters, temp_path=temp_path)
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.MASK_ON = True
        self.cfg.MODEL.KEYPOINT_ON = True
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.DATASETS.TEST = ()  
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = img_per_step
        self.cfg.SOLVER.BASE_LR = lr
        self.cfg.SOLVER.WEIGHT_DECAY = weight_decay
        self.cfg.SOLVER.GAMMA = gamma
        self.cfg.SOLVER.MOMENTUM = momentum
        self.cfg.SOLVER.MAX_ITER = iters
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size   
        self.cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = False
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = loss_weight
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
        self.cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((num_keypoints, 1), dtype=float).tolist()
        self.classes = ["RockerArm", "BoltHoles", "Big_PushRodHoles",
                        "Small_PushRodHoles", "Engine", "Bolt",
                        "PushRod", "RockerArmObject"]

        # Initialize temp path
        self.cfg.OUTPUT_DIR = temp_path
        if not os.path.exists(temp_path):
            os.makedirs(temp_path, exist_ok=True)
        
    def fit(self, dataset, val_dataset=None, verbose=True):
        self.__prepare_dataset(dataset)
        trainer = DefaultTrainer(self.cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()
        # return training_dict

    def __get_fiftyone_dicts(self, samples):
        samples.compute_metadata()

        dataset_dicts = []
        for sample in samples.select_fields(["id", "filepath", "metadata", "segmentations"]):
            height = sample.metadata["height"]
            width = sample.metadata["width"]
            record = {}
            record["file_name"] = sample.filepath
            record["image_id"] = sample.id
            record["height"] = height
            record["width"] = width

            objs = []
            for det in sample.segmentations.detections:
                tlx, tly, w, h = det.bounding_box
                bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
                fo_poly = det.to_polyline()
                poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
                poly = [p for x in poly for p in x]
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    def __prepare_dataset(self, dataset):
        # Split the dataset
        four.random_split(dataset, {"train": 0.8, "val": 0.2})
        # Register the dataset
        for d in ["train", "val"]:
            view = dataset.match_tags(d)
            DatasetCatalog.register("diesel_engine_" + d, lambda view=view: self.get_fiftyone_dicts(view))
            MetadataCatalog.get("diesel_engine_" + d).set(thing_classes=self.classes)
        return True

    def infer(self, img_data):
        if not isinstance(img_data, Image):
            img_data = Image(img_data)
        img_data = img_data.convert(format='channels_last', channel_order='rgb')
        output = self.predictor(img_data)
        pred_classes = output["instances"].to("cpu").pred_classes.numpy()
        bounding_boxes = output["instances"].to("cpu").pred_boxes.tensor.numpy()
        seg_masks = output["instances"].to("cpu").pred_masks.numpy()
        masks = seg_masks.astype('uint8')*255
        result = []
        for pred_class, bbox, seg_mask in zip(pred_classes, bounding_boxes, masks):
            result.append((BoundingBox(name=pred_class, left=bbox[0], top=bbox[1], width=bbox[2]-bbox[0], 
                          height=bbox[3]-bbox[1]), seg_mask))
        return result

    def load(self, model, verbose=True):
        assert os.path.isfile(model), "Checkpoint {} not found!".format(model)

        self.cfg.MODEL.WEIGHTS = str(model)      
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.classes)
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(self.classes)      
        self.predictor = DefaultPredictor(self.cfg)
        print("Model loaded!")

        if verbose:
            print("Loaded parameters and metadata.")
        return True

    def save(self, path, verbose=False):
        """ TODO """
        pass

    def download(self, path=None, mode="pretrained", verbose=False, 
                 url=OPENDR_SERVER_URL + "/perception/object_detection_2d/detectron2/"):
        """ TODO """
        pass

    def eval(self, json_file, image_root):
        dataset_name = "customValidationDataset"
        self.__prepare_dataset(dataset_name, json_file, image_root)
        self.cfg.DATASETS.TEST = (dataset_name,)
        output_folder = os.path.join(self.cfg.OUTPUT_DIR, "eval")
        evaluator = COCOEvaluator(dataset_name, self.cfg, False, output_folder)
        data_loader = build_detection_test_loader(self.cfg, dataset_name)
        inference_on_dataset(self.predictor.model, data_loader, evaluator)

    def optimize(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()

    def reset(self):
        """This method is not used in this implementation."""
        raise NotImplementedError()
