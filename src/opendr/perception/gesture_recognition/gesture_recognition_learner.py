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

import os
import datetime
import json
import zipfile
from pathlib import Path
import random
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ProgressBar

from opendr.perception.object_detection_2d.nanodet.nanodet_learner import NanodetLearner
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.arch import build_model
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.collate import naive_collate

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.evaluator import build_evaluator
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.trainer.task import TrainingTask

from opendr.perception.gesture_recognition.algorithm.data.dataset import build_dataset
from opendr.perception.gesture_recognition.algorithm.data.dataset.hagrid2coco import convert_to_coco

from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.datasets import ExternalDataset
from urllib.request import urlretrieve

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util import (
    cfg,
    load_config,
    mkdir,
    NanoDetLightningLogger,
)
_MODEL_NAMES = {"plus_m_1.5x_416"}


class GestureRecognitionLearner(NanodetLearner):
    def __init__(self, **kwargs):

        super(GestureRecognitionLearner, self).__init__(**kwargs)

        self.model = build_model(self.cfg.model)

    def preprocess_data(self, preprocess=True, download=False, verbose=True, save_path='./data/'):
        if download:
            if verbose:
                print('Downloading hagrid dataset....')
            main_url = "https://n-usr-2uzac.s3pd02.sbercloud.ru/b-usr-2uzac-mv4/hagrid/"
            test_urls = {"test": f"{main_url}test.zip",
                         "ann_train_val": f"{main_url}ann_train_val.zip", "ann_test": f"{main_url}ann_test.zip"}

            gestures = ["call", "dislike", "fist", "four", "like", "mute", "ok",
                        "one", "palm", "peace_inverted", "peace", "rock", "stop_inverted",
                        "stop", "three", "three2", "two_up_inverted", "two_up"]
            if verbose:
                print('Downloading annotations....')
            save_path_test = os.path.join(save_path, 'test')
            os.makedirs(save_path_test, exist_ok=True)

            os.system(f"wget {test_urls['ann_test']} -O {save_path}/ann_test.zip")
            os.system(f"wget {test_urls['ann_train_val']} -O {save_path}/ann_train_val.zip")

            with zipfile.ZipFile(os.path.join(save_path, "ann_test.zip"), 'r') as zip_ref:
                zip_ref.extractall(save_path)
            os.remove(os.path.join(save_path, "ann_test.zip"))
            with zipfile.ZipFile(os.path.join(save_path, "ann_train_val.zip"), 'r') as zip_ref:
                zip_ref.extractall(save_path)
            os.remove(os.path.join(save_path, "ann_train_val.zip"))

            if verbose:
                print('Downloading test data....')
            os.system(f"wget {test_urls['test']} -O {save_path_test}/test.zip")
            with zipfile.ZipFile(os.path.join(save_path_test, "test.zip"), 'r') as zip_ref:
                zip_ref.extractall(save_path_test)
            os.remove(os.path.join(save_path_test, "test.zip"))

            save_train = os.path.join(save_path, 'train')
            os.makedirs(save_train, exist_ok=True)

            save_val = os.path.join(save_path, 'val')
            os.makedirs(save_val, exist_ok=True)
            for target in gestures:
                if verbose:
                    print('Downloading {} class....'.format(target))
                target_url = main_url+"train_val_{}.zip".format(target)
                os.system(f"wget {target_url} -O {save_train}/{target}.zip")
                with zipfile.ZipFile(os.path.join(save_train, "{}.zip".format(target)), 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(save_train, target))
                n_val = int(0.2*len(os.listdir(os.path.join(save_train, target))))
                filenames = os.listdir(os.path.join(save_train, target))
                random.shuffle(filenames)
                val_files = filenames[:n_val]
                os.makedirs(os.path.join(save_val, target), exist_ok=True)
                for filename in val_files:
                    os.rename(os.path.join(save_train, target, filename), os.path.join(save_val, target, filename))

                os.remove(os.path.join(save_train, "{}.zip".format(target)))
        if preprocess:
            convert_to_coco(out=save_path, dataset_folder=save_path, dataset_annotations=save_path)
        dataset = ExternalDataset(save_path, 'coco')
        val_dataset = ExternalDataset(save_path, 'coco')
        test_dataset = ExternalDataset(save_path, 'coco')
        return dataset, val_dataset, test_dataset

    def fit(self, dataset, val_dataset, logging_path='', verbose=True, logging=False, seed=123, local_rank=1):
        """
        This method is used to train the gesture recognition model.
        :param dataset: training data
        :type dataset ExternalDataset
        :param val_dataset: training data
        :type val_dataset ExternalDataset
        :param logging_path: subdirectory in temp_path to save logger outputs
        :type logging_path: str
        :param verbose: if set to True, additional information is printed to STDOUT
        :type verbose: bool
        :param logging: if set to True, text and STDOUT logging will be used
        :type logging: bool
        :param seed: seed for reproducibility
        :type seed: int
        :param local_rank: for distribution learning
        :type local_rank: int
        """

        mkdir(local_rank, self.cfg.save_dir)

        if logging:
            self.logger = NanoDetLightningLogger(self.temp_path + "/" + logging_path)
            self.logger.dump_cfg(self.cfg)

        if seed != '' or seed is not None:
            if logging:
                self.logger.info("Set random seed to {}".format(seed))
            pl.seed_everything(seed)

        if logging:
            self.logger.info("Setting up data...")
        elif verbose:
            print("Setting up data...")

        train_dataset = build_dataset(self.cfg.data.train, dataset, self.cfg.class_names, "train")
        val_dataset = train_dataset if val_dataset is None else \
            build_dataset(self.cfg.data.val, val_dataset, self.cfg.class_names, "val")

        evaluator = build_evaluator(self.cfg.evaluator, val_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=False,
            collate_fn=naive_collate,
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=False,
            collate_fn=naive_collate,
            drop_last=False,
        )

        # Load state dictionary
        model_resume_path = (
            os.path.join(self.temp_path, "checkpoints", "model_iter_{}.ckpt".format(self.checkpoint_load_iter))
            if self.checkpoint_load_iter > 0 else None
        )

        if logging:
            self.logger.info("Creating task...")
        elif verbose:
            print("Creating task...")
        self.task = TrainingTask(self.cfg, self.model, evaluator)

        gpu_ids = None
        accelerator = None
        if self.device == "cuda":
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
            callbacks=[ProgressBar(refresh_rate=0)],
            logger=self.logger,
            benchmark=True,
            gradient_clip_val=self.cfg.get("grad_clip", 0.0),
        )

        trainer.fit(self.task, train_dataloader, val_dataloader)

    def eval(self, dataset, verbose=True, logging=False, local_rank=1):
        """
        This method performs evaluation on a given dataset and returns a dictionary with the evaluation results.
        :param dataset: test data
        :type dataset_path: ExternalDataset
        :param verbose: if set to True, additional information is printed to STDOUT
        :type verbose: bool
        :param logging: if set to True, text and STDOUT logging will be used
        :type logging: bool
        :param local_rank: for distribution learning
        :type local_rank: int
        """

        timestr = datetime.datetime.now().__format__("%Y_%m_%d_%H:%M:%S")
        save_dir = os.path.join(self.cfg.save_dir, timestr)
        mkdir(local_rank, save_dir)

        if logging:
            self.logger = NanoDetLightningLogger(save_dir)

        self.cfg.update({"test_mode": "val"})

        if logging:
            self.logger.info("Setting up data...")
        elif verbose:
            print("Setting up data...")

        val_dataset = build_dataset(self.cfg.data.val, dataset, self.cfg.class_names, "test")

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=False,
            collate_fn=naive_collate,
            drop_last=False,
        )
        evaluator = build_evaluator(self.cfg.evaluator, val_dataset)

        if logging:
            self.logger.info("Creating task...")
        elif verbose:
            print("Creating task...")

        self.task = TrainingTask(self.cfg, self.model, evaluator)

        gpu_ids = None
        accelerator = None
        if self.device == "cuda":
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
        if self.logger:
            self.logger.info("Starting testing...")
        elif verbose:
            print("Starting testing...")

        test_results = (verbose or logging)
        return trainer.test(self.task, val_dataloader, verbose=test_results)

    def download(self, path=None, verbose=True,
                 url=OPENDR_SERVER_URL + "/perception/gesture_recognition/nanodet/"):

        """
        Downloads model
        :param path: folder to which files will be downloaded, if None self.temp_path will be used
        :type path: str
        :param verbose: if True, additional information is printed on STDOUT
        :type verbose: bool
        :param url: URL to file location on FTP server
        :type url: str
        """

        if path is None:
            path = self.temp_path
        if not os.path.exists(path):
            os.makedirs(path)

        model = self.cfg.check_point_name

        path = os.path.join(path, "nanodet_{}".format(model))
        if not os.path.exists(path):
            os.makedirs(path)

        if verbose:
            print("Downloading pretrained checkpoint...")

        file_url = os.path.join(url, "nanodet_{}".format(model),
                                "nanodet_{}.ckpt".format(model))
        if not os.path.exists(os.path.join(path, "nanodet_{}.ckpt".format(model))):
            urlretrieve(file_url, os.path.join(path, "nanodet_{}.ckpt".format(model)))
        else:
            print("Checkpoint already exists.")

        if verbose:
            print("Downloading pretrain weights if provided...")

        file_url = os.path.join(url, "nanodet_{}".format(model),
                                     "nanodet_{}.pth".format(model))
        try:
            if not os.path.exists(os.path.join(path, "nanodet_{}.pth".format(model))):
                urlretrieve(file_url, os.path.join(path, "nanodet_{}.pth".format(model)))
            else:
                print("Weights file already exists.")

            if verbose:
                print("Making metadata...")
            metadata = {"model_paths": [], "framework": "pytorch", "format": "pth", "has_data": False,
                        "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                        "optimized": False, "optimizer_info": {}}

            param_filepath = "nanodet_{}.pth".format(model)
            metadata["model_paths"].append(param_filepath)
            with open(os.path.join(path, "nanodet_{}.json".format(model)), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

        except:
            print("Pretrain weights for this model are not provided!!! \n"
                  "Only the checkpoint will be download")

            if verbose:
                print("Making metadata...")
            metadata = {"model_paths": [], "framework": "pytorch", "format": "pth", "has_data": False,
                        "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                        "optimized": False, "optimizer_info": {}}

            param_filepath = "nanodet_{}.ckpt".format(model)
            metadata["model_paths"].append(param_filepath)
            with open(os.path.join(path, "nanodet_{}.json".format(model)), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

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
