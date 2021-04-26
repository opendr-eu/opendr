# Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import datetime
import json
import shutil
import random
import time
import warnings
import torch
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from urllib.request import urlretrieve

from perception.object_detection_2d.detr.algorithm.util.detect import detect
from perception.object_detection_2d.detr.algorithm.datasets import build_dataset, get_coco_api_from_dataset
from perception.object_detection_2d.detr.algorithm.engine import evaluate, train_one_epoch
from perception.object_detection_2d.detr.algorithm.models import build_model, build_criterion, build_postprocessors

from engine.constants import OPENDR_SERVER_URL
from engine.data import Image
from engine.learners import Learner
from engine.datasets import ExternalDataset, DatasetIterator
from engine.target import BoundingBox, BoundingBoxList

import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
import perception.object_detection_2d.detr.algorithm.util.misc as utils
from PIL import Image as im


class DetrLearner(Learner):
    def __init__(
        self,
        model_config_path=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "algorithm/configs/model_config.yaml"
            ),
        iters=10,
        lr=1e-4,
        batch_size=1,
        optimizer="adamw",
        lr_schedule="",
        backbone="resnet50",
        network_head="",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="temp",
        device="cuda",
        threshold=0.7,
        scale=1.0
    ):

        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(DetrLearner, self).__init__(
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

        # Initialise distributed mode in case of distributed mode
        utils.init_distributed_mode(self.args)
        if self.args.frozen_weights is not None:
            assert self.args.masks, "Frozen training is meant for segmentation only"

        # Fix the seed for reproducibility
        seed = self.args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Initialise epoch
        self.epoch = self.checkpoint_load_iter

        # Initialise transform for inference
        self.infer_transform = T.Compose([
            T.Resize(self.args.input_size),
            T.ToTensor(),
            T.Normalize(self.args.image_mean, self.args.image_std)
        ])

        # Initialise temp path
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        # Initialise ort
        self.ort_session = None

        # Initialize criterion, postprocessors, optimizer and scheduler
        self.criterion = None
        self.postprocessors = None
        self.torch_optimizer = None
        self.lr_scheduler = None
        self.n_parameters = None

    def save(self, path):
        """
        Method for saving the current model in the path provided.

        Parameters
        ----------
        path : str
            Folder where the model should be saved. If it does not exist, it
            will be created.

        Raises
        ------
        UserWarning
            If there is no model available, a warning is raised.

        Returns
        -------
        bool
            If True, model was saved was successfully.

        """
        if self.model is None and self.ort_session is None:
            raise UserWarning("No model is loaded, cannot save.")

        if not os.path.exists(path):
            os.makedirs(path)

        if self.ort_session is None:
            metadata = {'model_paths': os.path.join(path, "detr_" + self.backbone + '_model.pth'),
                        'framework': 'pytorch',
                        'format': 'pth',
                        'has_data': False,
                        'inference_params': {'threshold': self.threshold},
                        'optimized': False,
                        'optimizer_info': {}
                        }
            torch.save(self.model.state_dict(), metadata['model_paths'])
        else:
            metadata = {'model_paths': os.path.join(path, 'detr_' + self.backbone + '_model.onnx'),
                        'framework': 'pytorch',
                        'format': 'onnx',
                        'has_data': False,
                        'inference_params': {'threshold': self.threshold},
                        'optimized': True,
                        'optimizer_info': {}
                        }
            shutil.copy2(os.path.join(self.temp_path, "onnx_model_temp.onnx"),
                         metadata['model_paths'])

        with open(os.path.join(path, 'detr_' + self.backbone + '.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        return True

    def load(self, path):
        """
        Method for loading a model that was saved earlier.

        Parameters
        ----------
        path : str
            Folder where the model was saved.

        Raises
        ------
        UserWarning
            If the given folder does not exist, a warning is raised.

        Returns
        -------
        bool
            True if loading the model was successful.

        """
        if os.path.exists(os.path.join(path, 'detr_' + self.backbone + '.json')):
            with open(os.path.join(path, 'detr_' + self.backbone + '.json')) as f:
                metadata = json.load(f)
            self.threshold = metadata['inference_params']['threshold']
        else:
            raise UserWarning('No detr_' + self.backbone + '.json found. Please have a check')

        if metadata['optimized']:
            path_onnx = os.path.join(path, 'detr_' + self.backbone + '_model.onnx')
            if os.path.exists(path_onnx):
                self.ort_session = ort.InferenceSession(path_onnx)
                print("Loaded ONNX model.")
            else:
                raise UserWarning('No detr_' + self.backbone + '_model.onnx found. Please have a check')
        else:
            if os.path.exists(os.path.join(path, 'detr_' + self.backbone + '_model.pth')):
                self.__create_model()
                self.model_without_ddp.load_state_dict(torch.load(
                    os.path.join(path, 'detr_' + self.backbone + '_model.pth')))
                print("Loaded Pytorch model.")
            else:
                raise UserWarning('No detr_' + self.backbone + '_model.pth found. Please have a check')
        return True

    def __load_checkpoint(self, path):
        """
        Internal method for loading a checkpoint

        Parameters
        ----------
        path : str
            Path to the checkpoint.

        Raises
        ------
        e
            Error when provided path does not exist.

        Returns
        -------
        None.

        """
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except FileNotFoundError as e:
            e.strerror = path + " not found, " \
                "provided checkpoint_load_iter (" + \
                str(self.checkpoint_load_iter) + \
                ") doesn't correspond to a saved checkpoint.\nNo such file or directory."
            raise e
        self.model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            self.torch_optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.epoch = self.checkpoint_load_iter = checkpoint['epoch'] + 1
        print(f'Loaded checkpoint{self.checkpoint_load_iter:04}.pth')

    def fit(self, dataset, val_dataset=None, logging_path='', silent=False, verbose=True,
            annotations_folder='Annotations',
            train_images_folder='train2017',
            train_annotations_file='instances_train2017.json',
            val_images_folder='val2017',
            val_annotations_file='instances_val2017.json'
            ):
        """
        This method is used for training the algorithm on a train dataset and validating on a val dataset.

        Parameters
        ----------
        dataset : ExternalDataset class object or DatasetIterator class object
            Object that holds the training dataset.
        val_dataset : ExternalDataset class object or DatasetIterator class object, optional
            Object that holds the validation dataset. The default is None.
        logging_path : str, optional
            Path to save tensorboard log files. If set to None or '', tensorboard logging is
            disabled. The default is ''.
        silent : TYPE, optional
            If True, all printing of training progress reports and other information
            to STDOUT are disabled. The default is False.
        verbose : bool, optional
            Enables the maximum verbosity. The default is True.
        annotations_folder : str,
            Foldername of the annotations json file. This folder should be contained in the
            dataset path provided. The default is 'Annotations'.
        train_images_folder : str, optional
            Name of the folder that contains the train dataset images. This folder should be contained in
            the dataset path provided. Note that this is a folder name, not a path. The default is 'train2017'.
        train_annotations_file : str, optional
            Filename of the train annotations json file. This file should be contained in the
            dataset path provided. The default is 'instances_train2017.json'.
        val_images_folder : str, optional
            Folder name that contains the validation images. This folder should be contained
            in the dataset path provided. Note that this is a folder name, not a path. The default is 'val2017'.
        val_annotations_file : str, optional
            Filename of the validation annotations json file. This file should be
            contained in the dataset path provided in the annotations folder provided. The default is 'instances_val2017.json'.

        Returns
        -------
        dict
            Returns stats regarding the last evaluation ran.

        """
        if silent:
            verbose = False

        train_stats = {}
        test_stats = {}
        coco_evaluator = None

        if logging_path != '' and logging_path is not None:
            logging = True
            logging_dir = Path(logging_path)
        else:
            logging = False

        if self.model is None:
            self.__create_model()
            if not silent and verbose:
                print('number of params:', self.n_parameters)

        if self.postprocessors is None:
            self.__create_postprocessors()

        self.__create_criterion()

        self.__create_optimizer()

        self.__create_scheduler()

        if self.args.frozen_weights is not None:
            checkpoint = torch.load(self.args.frozen_weights, map_location=self.device)
            self.model_without_ddp.detr.load_state_dict(checkpoint['model'])

        if self.checkpoint_load_iter != 0:
            output_dir = Path(self.temp_path)
            checkpoint = output_dir / f'checkpoint{self.checkpoint_load_iter:04}.pth'
            self.__load_checkpoint(checkpoint)
            if not silent:
                print("Loaded" + f'checkpoint{self.checkpoint_load_iter:04}.pth')

        output_dir = Path(self.temp_path)
        device = torch.device(self.device)
        dataset_train = self.__prepare_dataset(
            dataset,
            image_set="train",
            images_folder_name=train_images_folder,
            annotations_folder_name=annotations_folder,
            annotations_file_name=train_annotations_file
            )

        if val_dataset is not None:
            dataset_val = self.__prepare_dataset(
                val_dataset,
                image_set="val",
                images_folder_name=val_images_folder,
                annotations_folder_name=annotations_folder,
                annotations_file_name=val_annotations_file
                )

        # Starting from here, code is mainly copied from https://github.com/facebookresearch/detr/blob/master/main.py
        # Main modifications:
        #   - Many variables are now attributes of the class
        #   - Added functionality for verbose and silent mode
        #   - Added possibibity to load from iteration specified by load_from_iter attribute

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
            data_loader_val = DataLoader(
                dataset_val,
                self.batch_size,
                sampler=sampler_val,
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=self.args.num_workers
                )
            base_ds = get_coco_api_from_dataset(dataset_val)

        if not silent:
            print("Start training")
        start_time = time.time()
        for self.epoch in range(self.checkpoint_load_iter, self.iters):
            if self.args.distributed:
                sampler_train.set_epoch(self.epoch)
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                data_loader_train,
                self.torch_optimizer,
                device,
                self.epoch,
                self.args.clip_max_norm,
                verbose=verbose,
                silent=silent
                )
            self.lr_scheduler.step()
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint every checkpoint_after_iter epochs
            if self.checkpoint_after_iter != 0 and (self.epoch + 1) % self.checkpoint_after_iter == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{self.epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': self.model_without_ddp.state_dict(),
                    'optimizer': self.torch_optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch': self.epoch,
                    'args': self.args,
                }, checkpoint_path)
            if val_dataset is not None:
                test_stats, coco_evaluator = evaluate(
                    self.model, self.criterion, self.postprocessors,
                    data_loader_val, base_ds, device, self.args.output_dir,
                    verbose=verbose, silent=silent)

            if logging:
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': self.epoch,
                    'n_parameters': self.n_parameters
                    }
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

        # End of code copied from https://github.com/facebookresearch/detr/blob/master/main.py

        if not silent:
            print('Training time {}'.format(total_time_str))
        if val_dataset is not None:
            return {train_stats, test_stats}
        return train_stats

    def eval(self,
             dataset,
             images_folder='val2017',
             annotations_folder='Annotations',
             annotations_file='instances_val2017.json'
             ):
        """
        This method is used to evaluate a trained model on an evaluation dataset.

        Parameters
        ----------
        dataset : ExternalDataset class object or DatasetIterator class object
            Object that holds the evaluation dataset.
        images_folder : str, optional
            Folder name that contains the dataset images. This folder should be contained in
            the dataset path provided. Note that this is a folder name, not a path. The default is 'val2017'.
        annotations_folder : str, optional
            Folder name of the annotations json file. This file should be contained in the
            dataset path provided. The default is 'Annotations'.
        annotations_file : str, optional
            Filename of the annotations json file. This file should be contained in the
            dataset path provided. The default is 'instances_val2017.json'.

        Raises
        ------
        UserWarning
            If there is no model, a warning is raised.

        Returns
        -------
        test_stats : dict
            Returns stats regarding evaluation.

        """
        if self.model is None:
            raise UserWarning('A model should be loaded first')

        if self.postprocessors is None:
            self.__create_postprocessors()

        self.__create_criterion()

        device = torch.device(self.device)

        dataset_val = self.__prepare_dataset(
            dataset,
            image_set="val",
            images_folder_name=images_folder,
            annotations_folder_name=annotations_folder,
            annotations_file_name=annotations_file
            )

        if self.args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = DataLoader(dataset_val, self.batch_size,
                                     sampler=sampler_val, drop_last=False,
                                     collate_fn=utils.collate_fn,
                                     num_workers=self.args.num_workers)

        base_ds = get_coco_api_from_dataset(dataset_val)

        test_stats, _ = evaluate(
                self.model, self.criterion, self.postprocessors,
                data_loader_val, base_ds, device,
                self.args.output_path
            )

        return test_stats

    def infer(self, image):
        """
        This method is used to perform object detection on an image.

        Parameters
        ----------
        image : engine.data.Image class object
            Image to run inference on.

        Returns
        -------
        engine.target.BoundingBoxList
            The engine.target.BoundingBoxList contains bounding boxes that are
            described by the left-top corner and its width and height, or
            returns an empty list if no detections were made.

        """
        if not isinstance(image, Image):
            image = Image(image)
        img = im.fromarray(image.numpy())

        scores, boxes = detect(img, self.infer_transform, self.model,
                               self.device, self.threshold, self.ort_session)

        boxlist = []
        for p, (xmin, ymin, xmax, ymax) in zip(
                scores.tolist(), boxes.tolist()):
            cl = np.argmax(p)
            name = f'{self.args.classes[cl]}'
            box = BoundingBox(name, xmin, ymin, xmax-xmin, ymax-ymin,
                              score=p[cl])
            boxlist.append(box)
        return BoundingBoxList(boxlist)

    def optimize(self, do_constant_folding=False):
        """
        Method for optimizing the model with onnx.

        Parameters
        ----------
        do_constant_folding : bool, optional
            If true, constant folding is true in the onnnx model. The default
            is False.

        Raises
        ------
        UserWarning
            If no model is loaded or if an ort session is already ongoing, a
            user warning is raised.

        Returns
        -------
        bool
            True if the model was optimized successfully.

        """
        if self.model is None:
            raise UserWarning("No model is loaded, cannot optimize. Load or train a model first.")

        if self.ort_session is not None:
            raise UserWarning("Model is already optimized in ONNX.")

        device = torch.device(self.device)

        x = torch.randn(1, 3, self.args.input_size[0],
                        self.args.input_size[1]).to(device)

        input_names = ['data']
        output_names = ['pred_logits', 'pred_boxes']

        torch.onnx.export(
            self.model,
            x,
            os.path.join(self.temp_path, "onnx_model_temp.onnx"),
            enable_onnx_checker=True,
            do_constant_folding=do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            opset_version=11
            )

        print("Exported onnx model")

        self.ort_session = ort.InferenceSession(
            os.path.join(self.temp_path, "onnx_model_temp.onnx")
            )
        return True

    def reset(self):
        """
        Method for resetting the state of the model. Since the model does not
        have a state, this method simply passes.

        Returns
        -------
        None.

        """

    def download_model(
            self, 
            panoptic=False, 
            backbone='resnet50', 
            dilation=False,
            pretrained=True,
            ):
        """
        Download utility for downloading detr models.

        Parameters
        ----------
        panoptic : bool, optional
            This bool differentiates between coco or coco_panoptic models.
            If set to false, the coco model is downloaded instead of the
            coco_panoptic model. The default is False.
        backbone : str, optional
            This str determines the backbone that is used in the model. There
            are two possible backbones: "resnet50" and "resnet101". The
            default is 'resnet50'.
        dilation : bool, optional
            If set to true, dilation is used in the model, otherwise not. The
            default is False.
        pretrained : bool, optional
            If set to true, a pretrained model is downloaded. The default is
            True.
        return_postprocessor : bool, optional
            If set to true, postprocessors are returned. The default is False.
        threshold : float, optional
            Sets the threshold for coco_panoptic models. The default is 0.85.

        Raises
        ------
        ValueError
            If an unsupported backbone is given, a value error is raised.

        Returns
        -------
        None.

        """
        torch.hub.set_dir(self.temp_path)

        supportedBackbones = ['resnet50', 'resnet101']
        if backbone.lower() in supportedBackbones:
            model_name = f'detr_{backbone}'
            self.backbone = self.args.backbone = backbone
            if dilation:
                model_name = model_name + '_dc5'
                self.args.dilation = True
            else:
                self.args.dilation = False
            if panoptic:
                model_name = model_name + '_panoptic'
                self.model = torch.hub.load(
                    'facebookresearch/detr',
                    model_name,
                    pretrained=pretrained,
                    num_classes=self.args.num_classes,
                    return_postprocessor=False,
                    threshold=self.threshold
                    )
                self.args.dataset_file = 'coco_panoptic'
            else:
                self.model = torch.hub.load(
                    'facebookresearch/detr',
                    model_name,
                    pretrained=pretrained,
                    num_classes=self.args.num_classes,
                    return_postprocessor=False
                    )
                self.args.dataset_file = 'coco'

            self.ort_session = None            

            device = torch.device(self.device)
            
            self.model.to(device)
            self.model_without_ddp = self.model

            if self.args.distributed:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu])
                self.model_without_ddp = self.model.module

            self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            raise ValueError(self.backbone + " not a valid backbone. Supported backbones:" + str(supportedBackbones))
    
    def download_nano_coco(self):
        url=OPENDR_SERVER_URL + "perception/object_detection_2d/detr/nano_coco/"
        
        if not os.path.exists(os.path.join(self.temp_path, "nano_coco")):
            os.makedirs(os.path.join(self.temp_path, "nano_coco"))
        
        if not os.path.exists(os.path.join(self.temp_path, "nano_coco", "image")):
            os.makedirs(os.path.join(self.temp_path, "nano_coco", "image"))
        
        # Download annotation file
        file_url = os.path.join(url, "instances.json")
        urlretrieve(file_url, os.path.join(self.temp_path, "nano_coco", "instances.json"))
        # Download test image
        file_url = os.path.join(url, "image", "000000391895.jpg")
        urlretrieve(file_url, os.path.join(self.temp_path, "nano_coco", "image", "000000391895.jpg"))
        
    
    def __create_criterion(self):
        """
        Internal model for creating the criterion.

        Returns
        -------
        None.

        """
        self.criterion = build_criterion(self.args)

    def __create_postprocessors(self):
        """
        Internal model for creating the postprocessors

        Returns
        -------
        None.

        """
        self.postprocessors = build_postprocessors(self.args)

    def __create_optimizer(self):
        """
        Internal model for creating the optimizer.

        Returns
        -------
        None.

        """
        param_dicts = [
            {"params": [p for n, p in self.model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.args.lr_backbone,
            },
        ]

        if self.optimizer == "adamw":
            self.torch_optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.optimizer == "adam":
            self.torch_optimizer = torch.optim.Adam(param_dicts, lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.optimizer == "sgd":
            self.torch_optimizer = torch.optim.SGD(param_dicts, lr=self.lr, weight_decay=self.args.weight_decay)
        else:
            warnings.warn("Unavailbale optimizer specified, using adamw instead. Possible optimizers are; adam, adamw and sgd")
            self.torch_optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def __create_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.torch_optimizer, self.args.lr_drop)

    def __create_model(self):
        """
        Internal method for creating a model, optimizer and scheduler based
        on the parameters in the config file.

        Returns
        -------
        None.

        """
        self.ort_session = None

        device = torch.device(self.device)
        self.model, self.criterion, self.postprocessors = build_model(self.args)
        self.model.to(device)

        self.model_without_ddp = self.model
        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu])
            self.model_without_ddp = self.model.module

        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __prepare_dataset(self,
                          dataset,
                          image_set="train",
                          images_folder_name="train2017",
                          annotations_folder_name="Annotations",
                          annotations_file_name="instances_train2017.json"
                          ):
        """
        This internal method prepares the dataset depending on what type of dataset is provided.

        If an ExternalDataset object type is provided, the method tried to prepare the dataset based on the original
        implementation, supposing that the dataset is in the COCO format. The path provided is searched for the
        images folder and the annotations file, converts the annotations file into the internal format used if needed
        and finally the CocoTrainDataset object is returned.

        If the dataset is of the DatasetIterator format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.

        Parameters
        ----------
        dataset : ExternalDataset class object or DatasetIterator class object
            The dataset.
        image_set : str, optional
            Specifies whether the dataset is a train or validation dataset, possible values: "train" or "val".
            The default is "train".
        images_folder_name : str, optional
            The name of the folder that contains the image files. The default is "train2017".
        annotations_folder_name : str, optional
            The folder that contains the original annotations. The default is "Annotations".
        annotations_file_name : str, optional
            The .json file that contains the original annotations. The default is "instances_train2017.json".

        Raises
        ------
        UserWarning
            UserWarnings with appropriate messages are raised for wrong type of dataset, or wrong paths
            and filenames.

        Returns
        -------
        CocoTrainDataset object or custom DatasetIterator implemented by user
            CocoDetection class object or DatasetIterator instance.

        """
        if not (isinstance(dataset, ExternalDataset) or isinstance(dataset, DatasetIterator)):
            raise UserWarning("dataset must be an ExternalDataset or a DatasetIterator")
        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() not in ["coco", "coco_panoptic"]:
                raise UserWarning("dataset_type must be \"COCO\" or \"COCO_PANOPTIC\"")

            # Get images folder
            images_folder = os.path.join(dataset.path, images_folder_name)
            if not os.path.isdir(images_folder):
                raise UserWarning("Didn't find \"" + images_folder_name +
                                  "\" folder in the dataset path provided.")

            # Get annotations file
            if annotations_folder_name == "":
                annotations_file = os.path.join(dataset.path, annotations_file_name)
                annotations_folder = dataset.path
            else:
                annotations_folder = os.path.join(dataset.path, annotations_folder_name)
                annotations_file = os.path.join(annotations_folder, annotations_file_name)
            if not os.path.isfile(annotations_file):
                raise UserWarning("Didn't find \"" + annotations_file +
                                  "\" file in the dataset path provided.")

            return build_dataset(images_folder, annotations_folder,
                                 annotations_file, image_set, self.args.masks,
                                 dataset.dataset_type.lower())
        return dataset
