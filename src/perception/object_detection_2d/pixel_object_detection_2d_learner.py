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
from util.detect import detect
from util.plot_utils import plot_results
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
        temp_path="/tmp",
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
        self.args.num_classes = len(self.args.classes)
        
        utils.init_distributed_mode(self.args)
        if self.args.frozen_weights is not None:
            assert self.args.masks, "Frozen training is meant for segmentation only"

        # fix the seed for reproducibility
        seed = self.args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.epoch = self.checkpoint_load_iter

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

    def load(self, path):
        """
        Loads a model from the path provided.

        :param path: Path to saved model
        :type path: str
        :return: Whether load succeeded or not
        :rtype: bool
        """
        
        if path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                path, map_location=self.device, check_hash=True)
        else:
            checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model'])
        # self.model.load_state_dict(checkpoint, strict=False)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.epoch = self.checkpoint_load_iter = checkpoint['epoch'] + 1
            
        return True

    def fit(self, dataset, val_dataset=None, logging_path='', 
            silent=False, verbose=True,
            annotations_folder='Annotations',
            train_images_folder='train2017',
            train_annotations_file='instances_train2017.json',
            val_images_folder='val2017',
            val_annotations_file='instances_val2017.json'):
        """
        This method is used for training the algorithm on a train dataset and validating on a val dataset.

        :param dataset: object that holds the training dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param val_dataset: object that holds the validation dataset, defaults to 'None'
        :type val_dataset: ExternalDataset class object or DatasetIterator class object, optional
        :param logging_path: path to save tensorboard log files. If set to None or '', tensorboard logging is
            disabled, defaults to ''
        :type logging_path: str, optional
        :param silent: if set to True, disables all printing of training progress reports and other information
            to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :param train_images_folder: folder name that contains the train dataset images. This folder should be contained in
            the dataset path provided. Note that this is a folder name, not a path, defaults to 'train2017'
        :type images_folder: str, optional
        :param annotations_folder: foldername of the annotations json file. This folder should be contained in the
            dataset path provided, defaults to 'Annotations'
        :type train_annotations_file: str, optional
        :param train_annotations_file: filename of the train annotations json file. This file should be contained in the
            dataset path provided, defaults to 'instances_train2017.json'
        :type train_annotations_file: str, optional
        :param val_images_folder: folder name that contains the validation images. This folder should be contained
            in the dataset path provided. Note that this is a folder name, not a path, defaults to 'val2017'
        :type val_images_folder_name: str, optional
        :param val_annotations_file: filename of the validation annotations json file. This file should be
            contained in the dataset path provided in the annotations folder provided, defaults to 'instances_val2017.json'
        :type val_annotations_file: str, optional

        :return: returns stats regarding the last evaluation ran
        :rtype: dict
        """
        train_stats = {}
        test_stats = {}
        coco_evaluator = None        

        if logging_path != '' and logging_path is not None:
            logging = True
            logging_dir = Path(logging_path)
        else: logging= False
        
        if self.model is None:
            self.__create_model()

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
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint every checkpoint_after_iter epochs
            if self.checkpoint_after_iter != 0 and (self.epoch + 1) % self.checkpoint_after_iter == 0:
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
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': self.epoch,
                     'n_parameters': self.n_parameters}
                with (logging_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None:
                    (logging_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if self.epoch % 50 == 0:
                            filenames.append(f'{self.epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       logging_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        return (train_stats, test_stats)
        
    def eval(self, dataset,
            images_folder='val2017',
            annotations_folder='Annotations',
            annotations_file='instances_val2017.json'):
        
        """
        This method is used to evaluate a trained model on an evaluation dataset.

        :param dataset: object that holds the evaluation dataset.
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param images_folde: Folder name that contains the dataset images. This folder should be contained in
            the dataset path provided. Note that this is a folder name, not a path, defaults to 'val2017'
        :type images_folder: str, optional
        :param annotations_folder: Folder name of the annotations json file. This file should be contained in the
            dataset path provided, defaults to 'Annotations'
        :type annotations_folder: str, optional
        :param annotations_file: Filename of the annotations json file. This file should be contained in the
            dataset path provided, defaults to 'pesron_keypoints_val2017.json'
        :type annotations_file: str, optional

        :returns: returns stats regarding evaluation
        :rtype: dict
        """
        
        device  = torch.device(self.device)
        
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
                data_loader_val, base_ds, device,
                self.args.output_path
            )

        return test_stats

    def infer(self, image):
        """
        This method is used to perform object detection on an image.

        :param img: image to run inference on
        :rtype img: engine.data.Image class object
        
        :return: Returns a engine.target.BoundingBoxList, 
        which contains bounding boxes that are described by the left-top corner 
        and its width and height, or returns an empty list if no
            detections were made.
        :rtype: engine.target.BoundingBoxList
        """
        
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scores, boxes = detect(image, transform, self.model, self. device)
        plot_results(image, scores, boxes, self.args.classes)
        

    def optimize(self, target_device):
        """
        This method optimizes the model based on the parameters provided.

        :param target_device: the optimization's procedure target device
        :type target_device: str
        :return: Whether optimize succeeded or not
        :rtype: bool
        """
        pass

    def reset(self):
        pass
    
    def download(self, panoptic=False, backbone='resnet50', dilation=False, 
                 pretrained=True, num_classes=91, return_postprocessor=False, 
                 threshold=0.85):
        
        supportedBackbones = ['resnet50', 'resnet101']
        if backbone.lower() in supportedBackbones:            
            model_name = f'detr_{backbone}'
            if dilation:
                model_name = model_name + '_dc5'
            if panoptic:
                model_name = model_name + '_panoptic'
                model = torch.hub.load('facebookresearch/detr', model_name, 
                                       pretrained=True, num_classes=num_classes, 
                                       return_postprocessor = return_postprocessor,
                                       threshold=threshold)
            else:
                model = torch.hub.load('facebookresearch/detr', model_name, 
                                       pretrained=True, num_classes=num_classes, 
                                       return_postprocessor = return_postprocessor)
            self.model = model
            self.model.to(torch.device(self.device))
        else:
            raise ValueError(self.backbone + " not a valid backbone. Supported backbones:" + str(supportedBackbones))
        
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
            
        if self.checkpoint_load_iter != 0:
            output_dir = Path(self.temp_path)
            checkpoint = output_dir / f'checkpoint{self.checkpoint_load_iter:04}.pth'
            self.load(path = checkpoint)

    def __prepare_dataset(self,
                          dataset,
                          image_set="train",
                          images_folder_name="train2017",
                          annotations_folder_name="Annotations",
                          annotations_file_name="instances_train2017.json"):
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
