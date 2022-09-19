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
import datetime
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ProgressBar

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.check_point import save_model_state
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.arch import build_model
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.collate import naive_collate
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.dataset import build_dataset
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.trainer.task import TrainingTask
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.evaluator import build_evaluator
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.inferencer.utilities import Predictor
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util import (
    NanoDetLightningLogger,
    Logger,
    cfg,
    load_config,
    load_model_weight,
    mkdir,
)

from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.engine.constants import OPENDR_SERVER_URL

from opendr.engine.learners import Learner
from urllib.request import urlretrieve

_MODEL_NAMES = {"EfficientNet_Lite0_320", "EfficientNet_Lite1_416", "EfficientNet_Lite2_512",
                "RepVGG_A0_416", "t", "g", "m", "m_416", "m_0.5x", "m_1.5x", "m_1.5x_416",
                "plus_m_320", "plus_m_1.5x_320", "plus_m_416", "plus_m_1.5x_416", "custom"}


class NanodetLearner(Learner):
    def __init__(self, model_to_use="plus_m_1.5x_416", iters=None, lr=None, batch_size=None, checkpoint_after_iter=None,
                 checkpoint_load_iter=None, temp_path='', device='cuda', weight_decay=None, warmup_steps=None,
                 warmup_ratio=None, lr_schedule_T_max=None, lr_schedule_eta_min=None, grad_clip=None):

        """Initialise the Nanodet Learner"""

        self.cfg = self._load_hparam(model_to_use)
        self.lr_schedule_T_max = lr_schedule_T_max
        self.lr_schedule_eta_min = lr_schedule_eta_min
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.grad_clip = grad_clip

        self.overwrite_config(lr=lr, weight_decay=weight_decay, iters=iters, batch_size=batch_size,
                              checkpoint_after_iter=checkpoint_after_iter, checkpoint_load_iter=checkpoint_load_iter,
                              temp_path=temp_path)

        self.lr = float(self.cfg.schedule.optimizer.lr)
        self.weight_decay = float(self.cfg.schedule.optimizer.weight_decay)
        self.iters = int(self.cfg.schedule.total_epochs)
        self.batch_size = int(self.cfg.device.batchsize_per_gpu)
        self.temp_path = self.cfg.save_dir
        self.checkpoint_after_iter = int(self.cfg.schedule.val_intervals)
        self.checkpoint_load_iter = int(self.cfg.schedule.resume)
        self.device = device
        self.classes = self.cfg.class_names

        super(NanodetLearner, self).__init__(lr=self.lr, iters=self.iters, batch_size=self.batch_size,
                                             checkpoint_after_iter=self.checkpoint_after_iter,
                                             checkpoint_load_iter=self.checkpoint_load_iter,
                                             temp_path=self.temp_path, device=self.device)

        self.model = build_model(self.cfg.model)
        self.logger = None
        self.task = None

    def _load_hparam(self, model: str):
        """ Load hyperparameters for nanodet models and training configuration

        :parameter model: The name of the model of which we want to load the config file
        :type model: str
        :return: config with hyperparameters
        :rtype: dict
        """
        assert (
                model in _MODEL_NAMES
        ), f"Invalid model selected. Choose one of {_MODEL_NAMES}."
        full_path = list()
        path = Path(__file__).parent / "algorithm" / "config"
        wanted_file = "nanodet_{}.yml".format(model)
        for root, dir, files in os.walk(path):
            if wanted_file in files:
                full_path.append(os.path.join(root, wanted_file))
        assert (len(full_path) == 1), f"You must have only one nanodet_{model}.yaml file in your config folder"
        load_config(cfg, full_path[0])
        return cfg

    def overwrite_config(self, lr=0.001, weight_decay=0.05, iters=10, batch_size=64, checkpoint_after_iter=0,
                         checkpoint_load_iter=0, temp_path=''):
        """
        Helping method for config file update to overwrite the cfg with arguments of OpenDR.
        :param lr: learning rate used in training
        :type lr: float, optional
        :param weight_decay: weight_decay used in training
        :type weight_decay: float, optional
        :param iters: max epoches that the training will be run
        :type iters: int, optional
        :param batch_size: batch size of each gpu in use, if device is cpu batch size
         will be used one single time for training
        :type batch_size: int, optional
        :param checkpoint_after_iter: after that number of epoches, evaluation will be
         performed and one checkpoint will be saved
        :type checkpoint_after_iter: int, optional
        :param checkpoint_load_iter: the epoch in which checkpoint we want to load
        :type checkpoint_load_iter: int, optional
        :param temp_path: path to a temporal dictionary for saving models, logs and tensorboard graphs.
         If temp_path='' the `cfg.save_dir` will be used instead.
        :type temp_path: str, optional
        """
        self.cfg.defrost()

        # Nanodet specific parameters
        if self.cfg.model.arch.head.num_classes != len(self.cfg.class_names):
            raise ValueError(
                "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
                "but got {} and {}".format(
                    self.cfg.model.arch.head.num_classes, len(self.cfg.class_names)
                )
            )
        if self.warmup_steps is not None:
            self.cfg.schedule.warmup.warmup_steps = self.warmup_steps
        if self.warmup_ratio is not None:
            self.cfg.schedule.warmup.warmup_ratio = self.warmup_ratio
        if self.lr_schedule_T_max is not None:
            self.cfg.schedule.lr_schedule.T_max = self.lr_schedule_T_max
        if self.lr_schedule_eta_min is not None:
            self.cfg.schedule.lr_schedule.eta_min = self.lr_schedule_eta_min
        if self.grad_clip is not None:
            self.cfg.grad_clip = self.grad_clip

        # OpenDR
        if lr is not None:
            self.cfg.schedule.optimizer.lr = lr
        if weight_decay is not None:
            self.cfg.schedule.optimizer.weight_decay = weight_decay
        if iters is not None:
            self.cfg.schedule.total_epochs = iters
        if batch_size is not None:
            self.cfg.device.batchsize_per_gpu = batch_size
        if checkpoint_after_iter is not None:
            self.cfg.schedule.val_intervals = checkpoint_after_iter
        if checkpoint_load_iter is not None:
            self.cfg.schedule.resume = checkpoint_load_iter
        if temp_path != '':
            self.cfg.save_dir = temp_path

        self.cfg.freeze()

    def save(self, path=None, verbose=True):
        """
        Method for saving the current model and metadata in the path provided.
        :param path: path to folder where model will be saved
        :type path: str, optional
        :param verbose: whether to print a success message or not, defaults to False
        :type verbose: bool, optional
        """
        path = path if path is not None else self.cfg.save_dir
        model = self.cfg.check_point_name
        os.makedirs(path, exist_ok=True)

        metadata = {"model_paths": [], "framework": "pytorch", "format": "pth",
                    "has_data": False, "inference_params": {}, "optimized": False,
                    "optimizer_info": {}, "classes": self.classes}

        param_filepath = "nanodet_{}.pth".format(model)
        metadata["model_paths"].append(param_filepath)

        logger = self.logger if verbose else None
        if self.task is None:
            print("You do not have call a task yet, only the state of the loaded or initialized model will be saved")
            save_model_state(os.path.join(path, metadata["model_paths"][0]), self.model, None, logger)
        else:
            self.task.save_current_model(os.path.join(path, metadata["model_paths"][0]), logger)

        with open(os.path.join(path, "nanodet_{}.json".format(model)), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        if verbose:
            print("Model metadata saved.")
        return True

    def load(self, path=None, verbose=True):
        """
        Loads the model from the path provided.
        :param path: path of the directory where the model was saved
        :type path: str, optional
        :param verbose: whether to print a success message or not, defaults to False
        :type verbose: bool, optional
        """
        path = path if path is not None else self.cfg.save_dir
        model = self.cfg.check_point_name
        if verbose:
            print("Model name:", model, "-->", os.path.join(path, model + ".json"))
        with open(os.path.join(path, "nanodet_{}.json".format(model))) as f:
            metadata = json.load(f)

        logger = Logger(-1, path, False) if verbose else None
        ckpt = torch.load(os.path.join(path, metadata["model_paths"][0]), map_location=torch.device(self.device))
        self.model = load_model_weight(self.model, ckpt, logger)
        if verbose:
            logger.log("Loaded model weight from {}".format(path))
        pass

    def download(self, path=None, mode="pretrained", verbose=False,
                 url=OPENDR_SERVER_URL + "/perception/object_detection_2d/nanodet/"):

        """
        Downloads all files necessary for inference, evaluation and training. Valid mode options are: ["pretrained",
        "images", "test_data"].
        :param path: folder to which files will be downloaded, if None self.temp_path will be used
        :type path: str, optional
        :param mode: one of: ["pretrained", "images", "test_data"], where "pretrained" downloads a pretrained
        network depending on the network choosed in config file, "images" downloads example inference data,
        and "test_data" downloads additional image,annotation file and pretrained network for training and testing
        :type mode: str, optional
        :param model: the specific name of the model to download, all pre-configured configs files have their pretrained
        model and can be selected, if None self.cfg.check_point_name will be used
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional
        :param url: URL to file location on FTP server
        :type url: str, optional
        """

        valid_modes = ["pretrained", "images", "test_data"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.temp_path
        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":

            model = self.cfg.check_point_name

            path = os.path.join(path, "nanodet_{}".format(model))
            if not os.path.exists(path):
                os.makedirs(path)

            if verbose:
                print("Downloading pretrained checkpoint...")

            file_url = os.path.join(url, "pretrained",
                                    "nanodet_{}".format(model),
                                    "nanodet_{}.ckpt".format(model))

            urlretrieve(file_url, os.path.join(path, "nanodet_{}.ckpt".format(model)))

            if verbose:
                print("Downloading pretrain weights if provided...")

            file_url = os.path.join(url, "pretrained", "nanodet_{}".format(model),
                                    "nanodet_{}.pth".format(model))
            try:
                urlretrieve(file_url, os.path.join(path, "nanodet_{}.pth".format(model)))

                if verbose:
                    print("Making metadata...")
                metadata = {"model_paths": [], "framework": "pytorch", "format": "pth",
                            "has_data": False, "inference_params": {}, "optimized": False,
                            "optimizer_info": {}, "classes": self.classes}

                param_filepath = "nanodet_{}.pth".format(model)
                metadata["model_paths"].append(param_filepath)
                with open(os.path.join(path, "nanodet_{}.json".format(model)), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

            except:
                print("Pretrain weights for this model are not provided!!! \n"
                      "Only the hole ckeckpoint will be download")

                if verbose:
                    print("Making metadata...")
                metadata = {"model_paths": [], "framework": "pytorch", "format": "pth",
                            "has_data": False, "inference_params": {}, "optimized": False,
                            "optimizer_info": {}, "classes": self.classes}

                param_filepath = "nanodet_{}.ckpt".format(model)
                metadata["model_paths"].append(param_filepath)
                with open(os.path.join(path, "nanodet_{}.json".format(model)), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

        elif mode == "images":
            file_url = os.path.join(url, "images", "000000000036.jpg")
            if verbose:
                print("Downloading example image...")
            urlretrieve(file_url, os.path.join(path, "000000000036.jpg"))

        elif mode == "test_data":
            os.makedirs(os.path.join(path, "test_data"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "train"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "val"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "train", "JPEGImages"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "train", "Annotations"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "val", "JPEGImages"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "val", "Annotations"), exist_ok=True)
            # download image
            file_url = os.path.join(url, "images", "000000000036.jpg")
            if verbose:
                print("Downloading image...")
            urlretrieve(file_url, os.path.join(path, "test_data", "train", "JPEGImages", "000000000036.jpg"))
            urlretrieve(file_url, os.path.join(path, "test_data", "val", "JPEGImages", "000000000036.jpg"))
            # download annotations
            file_url = os.path.join(url, "annotations", "000000000036.xml")
            if verbose:
                print("Downloading annotations...")
            urlretrieve(file_url, os.path.join(path, "test_data", "train", "Annotations", "000000000036.xml"))
            urlretrieve(file_url, os.path.join(path, "test_data", "val", "Annotations", "000000000036.xml"))

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def optimize(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def fit(self, dataset, val_dataset=None, logging_path='', verbose=True, seed=123):
        """
        This method is used to train the detector on the COCO dataset. Validation is performed in a val_dataset if
        provided, else validation is performed in training dataset.
        :param dataset: training dataset; COCO and Pascal VOC are supported as ExternalDataset types,
        with 'coco' or 'voc' dataset_type attributes. custom DetectionDataset types are not supported at the moment.
        Any xml type dataset can be use if voc is used in datatype.
        :type dataset: ExternalDataset, DetectionDataset not implemented yet
        :param val_dataset: validation dataset object
        :type val_dataset: ExternalDataset, DetectionDataset not implemented yet
        :param logging_path: subdirectory in temp_path to save logger outputs
        :type logging_path: str, optional
        :param verbose: if set to True, additional information is printed to STDOUT and logger txt output,
        defaults to True
        :type verbose: bool
        :param seed: seed for reproducibility
        :type seed: int
        """

        mkdir(self.cfg.save_dir)

        if verbose:
            self.logger = NanoDetLightningLogger(self.temp_path + "/" + logging_path)
            self.logger.dump_cfg(self.cfg)

        if seed != '' or seed is not None:
            if verbose:
                self.logger.info("Set random seed to {}".format(seed))
            pl.seed_everything(seed)

        if verbose:
            self.logger.info("Setting up data...")

        train_dataset = build_dataset(self.cfg.data.val, dataset, self.cfg.class_names, "train")
        val_dataset = train_dataset if val_dataset is None else \
            build_dataset(self.cfg.data.val, val_dataset, self.cfg.class_names, "val")

        evaluator = build_evaluator(self.cfg.evaluator, val_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=False,
        )

        # Load state dictionary
        model_resume_path = (
            os.path.join(self.temp_path, "checkpoints", "model_iter_{}.ckpt".format(self.checkpoint_load_iter))
            if self.checkpoint_load_iter > 0 else None
        )

        if verbose:
            self.logger.info("Creating task...")
        self.task = TrainingTask(self.cfg, self.model, evaluator)

        if self.device == "cpu":
            gpu_ids = None
            accelerator = None
        elif self.device == "cuda":
            gpu_ids = self.cfg.device.gpu_ids
            accelerator = None if len(gpu_ids) <= 1 else "ddp"

        trainer = pl.Trainer(
            default_root_dir=self.temp_path,
            max_epochs=self.iters,
            gpus=gpu_ids,
            check_val_every_n_epoch=self.checkpoint_after_iter,
            accelerator=accelerator,
            log_every_n_steps=self.cfg.log.interval,
            num_sanity_val_steps=0,
            resume_from_checkpoint=model_resume_path,
            callbacks=[ProgressBar(refresh_rate=0)],  # disable tqdm bar
            logger=self.logger,
            benchmark=True,
            gradient_clip_val=self.cfg.get("grad_clip", 0.0),
        )

        trainer.fit(self.task, train_dataloader, val_dataloader)

    def eval(self, dataset, verbose=True):
        """
        This method performs evaluation on a given dataset and returns a dictionary with the evaluation results.
        :param dataset: dataset object, to perform evaluation on
        :type dataset: ExternalDataset, DetectionDataset not implemented yet
        :param verbose: if set to True, additional information is printed to STDOUT and logger txt output,
        defaults to True
        :type verbose: bool
        """

        timestr = datetime.datetime.now().__format__("%Y_%m_%d_%H:%M:%S")
        save_dir = os.path.join(self.cfg.save_dir, timestr)
        mkdir(save_dir)

        if verbose:
            self.logger = NanoDetLightningLogger(save_dir)

        self.cfg.update({"test_mode": "val"})

        if verbose:
            self.logger.info("Setting up data...")

        val_dataset = build_dataset(self.cfg.data.val, dataset, self.cfg.class_names, "val")

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=False,
        )
        evaluator = build_evaluator(self.cfg.evaluator, val_dataset)

        if verbose:
            self.logger.info("Creating task...")
        self.task = TrainingTask(self.cfg, self.model, evaluator)

        if self.device == "cpu":
            gpu_ids = None
            accelerator = None
        elif self.device == "cuda":
            gpu_ids = self.cfg.device.gpu_ids
            accelerator = None if len(gpu_ids) <= 1 else "ddp"

        trainer = pl.Trainer(
            default_root_dir=save_dir,
            gpus=gpu_ids,
            accelerator=accelerator,
            log_every_n_steps=self.cfg.log.interval,
            num_sanity_val_steps=0,
            logger=self.logger,
        )
        if verbose:
            self.logger.info("Starting testing...")
        return trainer.test(self.task, val_dataloader, verbose=verbose)

    def infer(self, input, threshold=0.35, verbose=True):
        """
        Performs inference
        :param input: input can be an Image type image to perform inference
        :type input: str, optional
        :param threshold: confidence threshold
        :type threshold: float, optional
        :param verbose: if set to True, additional information is printed to STDOUT and logger txt output,
        defaults to True
        :type verbose: bool
        :return: list of bounding boxes of last image of input or last frame of the video
        :rtype: BoundingBoxList
        """

        if verbose:
            self.logger = Logger(0, use_tensorboard=False)
        predictor = Predictor(self.cfg, self.model, device=self.device)
        if not isinstance(input, Image):
            input = Image(input)
        _input = input.opencv()
        meta, res = predictor.inference(_input, verbose)

        bounding_boxes = BoundingBoxList([])
        for label in res[0]:
            for box in res[0][label]:
                score = box[-1]
                if score > threshold:
                    bbox = BoundingBox(left=box[0], top=box[1],
                                       width=box[2] - box[0],
                                       height=box[3] - box[1],
                                       name=label,
                                       score=score)
                    bounding_boxes.data.append(bbox)
        bounding_boxes.data.sort(key=lambda v: v.confidence)

        return bounding_boxes
