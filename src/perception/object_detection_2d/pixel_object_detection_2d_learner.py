# Copyright 2020 Delft University of Technology
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
import random
import time
import warnings
import torchvision.transforms as T
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from detr_engine import evaluate, train_one_epoch
from models import build_model

from engine.learners import Learner
from engine.datasets import ExternalDataset, DatasetIterator

class PixelObjectDetection2DLearner(Learner):
    def __init__(
        self,
        model_config_path=os.path.join(
            os.path.dirname(os.path.realpath('__file__')),
            "configs/model_config.yaml"),
        iters=10,
        lr=1e-4,
        batch_size=1,
        optimizer="adamw",
        lr_schedule="",
        backbone="resnet50",
        network_head="",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="/temp",
        device="cuda",
        threshold=0.0,
        scale=1.0
    ):

        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(PixelObjectDetection2DLearner, self).__init__(
            iters=iters,
            lr=lr,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            backbone=backbone,
            network_head=network_head,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            temp_path=temp_path,
            device=device,
            threshold=threshold,
            scale=scale,
        )

        # Add arguments to a structure like in the original implementation


        self.args = utils.load_config(model_config_path)
        self.args.lr = self.lr
        self.args.batch_size = self.batch_size
        self.args.optimizer = self.optimizer
        self.args.backbone = self.backbone
        self.args.device = self.device
        self.args.output_path = self.temp_path

        utils.init_distributed_mode(self.args)
        if self.args.frozen_weights is not None:
            assert self.args.masks, "Frozen training is meant for segmentation only"

        # fix the seed for reproducibility
        seed = self.args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.epoch = self.checkpoint_load_iter
        self.__create_model()

    def save(self, path=''):
        """
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str
        :return: Whether save succeeded or not
        :rtype: bool
        """

        checkpoint_path = os.path.join(path,'checkpoint.pth')
        utils.save_on_master({'model': self.model_without_ddp.state_dict(),
                              'optimizer': self.optim.state_dict(),
                              'lr_scheduler': self.lr_scheduler.state_dict(),
                              'epoch': self.epoch,
                              'args': self.args,
                              }, checkpoint_path)

        return True

    def load(self, path="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"):
        """
        Loads the model from the specified path or url.
        """
        if path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                path, map_location=self.device, check_hash=True)
        else:
            checkpoint = torch.load(path, map_location="cpu")
        self.model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.checkpoint_load_iter = checkpoint['epoch'] + 1

    def reset(self):
        pass

    def fit(self, dataset, val_dataset=None,
            annotations_folder='Annotations',
            train_images_folder='train2017',
            train_annotations_file='instances_train2017.json',
            val_images_folder='val2017',
            val_annotations_file='instances_val2017.json',
            logging_path='', silent=False, verbose=False):

        if logging_path != '' and logging_path is not None:
            logging = True
            logging_dir = Path(logging_path)
        else: logging= False

        output_dir = Path(self.temp_path)
        device = torch.device(self.device)

        dataset_train = self.__prepare_dataset(dataset,
                                      image_set="train",
                                      images_folder_name=train_images_folder,
                                      annotations_folder_name=annotations_folder,
                                      annotations_file_name=train_annotations_file)

        if val_dataset is not None:
            dataset_val = self.__prepare_dataset(val_dataset,
                                      image_set="val",
                                      images_folder_name=val_images_folder,
                                      annotations_folder_name=annotations_folder,
                                      annotations_file_name=val_annotations_file)

        if self.args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            if val_dataset is not None:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            if val_dataset is not None:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)


        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, self.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn,
                                       num_workers=self.args.num_workers)
        if val_dataset is not None:
            data_loader_val = DataLoader(dataset_val, self.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=self.args.num_workers)
            base_ds = get_coco_api_from_dataset(dataset_val)



        print("Start training")
        start_time = time.time()
        for self.epoch in range(self.checkpoint_load_iter, self.iters):
            if self.args.distributed:
                sampler_train.set_epoch(self.epoch)
            train_stats = train_one_epoch(self.model, self.criterion, data_loader_train, self.optim, device, self.epoch,
                self.args.clip_max_norm)
            self.lr_scheduler.step()
            if self.checkpoint_after_iter !=0:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (self.epoch + 1) % self.args.lr_drop == 0 or (self.epoch + 1) % self.checkpoint_after_iter == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{self.epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': self.model_without_ddp.state_dict(),
                        'optimizer': self.optim.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict(),
                        'epoch': self.epoch,
                        'args': self.args,
                    }, checkpoint_path)
            if val_dataset is not None:
                test_stats, coco_evaluator = evaluate(
                    self.model, self.criterion, self.postprocessors,
                    data_loader_val, base_ds, device, self.args.output_dir
                    )

            if logging:
                if val_dataset is not None:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': self.epoch,
                         'n_parameters': self.n_parameters}
                    with (logging_dir / "log.txt").open("a") as f:
                            f.write(json.dumps(log_stats) + "\n")

                    (logging_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if self.epoch % 50 == 0:
                            filenames.append(f'{self.epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)
                else:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': self.epoch,
                         'n_parameters': self.n_parameters}
                    with (logging_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def eval(self, dataset, logging_path=None, output_path=None,
            annotations_folder='Annotations',
            images_folder='val2017',
            annotations_file='instances_val2017.json'):

        if logging_path != '' and logging_path is not None:
            logging = True
            logging_dir = Path(logging_path)

        dataset_val = self.__prepare_dataset(dataset,
                                      image_set="val",
                                      images_folder_name=images_folder,
                                      annotations_folder_name=annotations_folder,
                                      annotations_file_name=annotations_file)

        if self.args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = DataLoader(dataset_val, self.batch_size,
                                     sampler=sampler_val, drop_last=False,
                                     collate_fn=utils.collate_fn,
                                     num_workers=self.args.num_workers)

        base_ds = get_coco_api_from_dataset(dataset_val)

        test_stats, coco_evaluator = evaluate(
                self.model, self.criterion, self.postprocessors,
                data_loader_val, base_ds, self.torch_device,
                self.args.output_path
            )
        if output_path is not None:
            output_dir = Path(output_path)
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        if logging:
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': self.epoch,
                 'n_parameters': self.n_parameters}
            with (logging_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            (logging_dir / 'eval').mkdir(exist_ok=True)
            if "bbox" in coco_evaluator.coco_eval:
                filenames = ['latest.pth']
                if self.epoch % 50 == 0:
                    filenames.append(f'{self.epoch:03}.pth')
                for name in filenames:
                    torch.save(coco_evaluator.coco_eval["bbox"].eval,
                               output_dir / "eval" / name)
        return test_stats

    def infer(self):
        pass

    def optimize(self):
        pass

    def __create_model(self):
        device = torch.device(self.device)
        self.model, self.criterion, self.postprocessors = build_model(self.args)
        self.model.to(device)

        self.model_without_ddp = self.model
        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu])
            self.model_without_ddp = self.model.module
        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', self.n_parameters)

        param_dicts = [
            {"params": [p for n, p in self.model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.args.lr_backbone,
            },
        ]

        if self.optimizer == "adamw":
            self.optim = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.optimizer == "adam":
            self.optim = torch.optim.Adam(param_dicts, lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.optimizer == "sgd":
            self.optim = torch.optim.SGD(param_dicts, lr=self.lr, weight_decay=self.args.weight_decay)
        else:
            warnings.warn("Unavailbale optimizer specified, using adamw instead. Possible optimizers are; adam, adamw and sgd")
            self.optim = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.lr_drop)

        if self.args.frozen_weights is not None:
            checkpoint = torch.load(self.args.frozen_weights, map_location=self.device)
            self.model_without_ddp.detr.load_state_dict(checkpoint['model'])

    def __prepare_dataset(self,
                          dataset,
                          image_set="train",
                          images_folder_name="train2017",
                          annotations_folder_name="Annotations",
                          annotations_file_name="instances_train2017.json",
                          verbose=True):
        """
        This internal method prepares the dataset depending on what type of dataset is provided.

        If an ExternalDataset object type is provided, the method tried to prepare the dataset based on the original
        implementation, supposing that the dataset is in the COCO format. The path provided is searched for the
        images folder and the annotations file, converts the annotations file into the internal format used if needed
        and finally the CocoTrainDataset object is returned.

        If the dataset is of the DatasetIterator format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.

        :param dataset: the dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param image_set: Specifies whether the dataset is a train or validation dataset, possible values: "train" or "val"
        :type image_set: str
        :param images_folder_name: the name of the folder that contains the image files, defaults to "train2017"
        :type images_folder_name: str, optional
        :param annotations_folder_name: the folder that contains the original annotations, defaults
            to "Annotations"
        :type annotations_folder_name: str, optional
        :param annotations_file_name: the .json file that contains the original annotations, defaults
            to "instances_train2017.json"
        :type annotations_file_name: str, optional
        :param verbose: whether to print additional information, defaults to 'True'
        :type verbose: bool, optional

        :raises UserWarning: UserWarnings with appropriate messages are raised for wrong type of dataset, or wrong paths
            and filenames

        :return: returns CocoTrainDataset object or custom DatasetIterator implemented by user
        :rtype: CocoDetection class object or DatasetIterator instance
        """

        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() not in ["coco", "coco_panoptic"]:
                raise UserWarning("dataset_type must be \"COCO\" or \"COCO_PANOPTIC\"")

            # Get images folder
            images_folder = os.path.join(dataset.path, images_folder_name)
            if not(os.path.isdir(images_folder)):
                raise UserWarning("Didn't find \"" + images_folder_name +
                                  "\" folder in the dataset path provided.")

            # Get annotations file
            if annotations_folder_name == "":
                annotations_file = os.path.join(dataset.path, annotations_file_name)
                annotations_folder = dataset.path
            else:
                annotations_folder = os.path.join(dataset.path, annotations_folder_name)
                annotations_file = os.path.join(annotations_folder, annotations_file_name)
            if not(os.path.isfile(annotations_file)):
                raise UserWarning("Didn't find \"" + annotations_file +
                                  "\" file in the dataset path provided.")

            return build_dataset(images_folder, annotations_folder,
                                 annotations_file, image_set, self.args.masks,
                                 dataset.dataset_type.lower())
        elif isinstance(dataset, DatasetIterator):
            return dataset
