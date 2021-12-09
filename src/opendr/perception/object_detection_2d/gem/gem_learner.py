# Copyright 2020-2021 OpenDR European Project

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
import torch
import ntpath
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from urllib.request import urlretrieve

from opendr.perception.object_detection_2d.detr.algorithm.datasets import get_coco_api_from_dataset
from opendr.perception.object_detection_2d.detr.algorithm.datasets.coco import map_bounding_box_list_to_coco
from opendr.perception.object_detection_2d.gem.algorithm.util.detect import detect
from opendr.perception.object_detection_2d.gem.algorithm.util.sampler import (RandomSampler, SequentialSampler,
                                                                              DistributedSamplerWrapper)
from opendr.perception.object_detection_2d.gem.algorithm.datasets import build_dataset
from opendr.perception.object_detection_2d.gem.algorithm.engine import evaluate, train_one_epoch
from opendr.perception.object_detection_2d.gem.algorithm.models import build_model, build_criterion, \
    build_postprocessors

from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.data import Image
from opendr.engine.learners import Learner
from opendr.engine.datasets import ExternalDataset, DatasetIterator, MappedDatasetIterator
from opendr.engine.target import CocoBoundingBox, BoundingBoxList

import torchvision.transforms as T
import numpy as np
import opendr.perception.object_detection_2d.detr.algorithm.util.misc as utils
from PIL import Image as im

import zipfile


class GemLearner(Learner):
    def __init__(
            self,
            model_config_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "algorithm/configs/model_config.yaml"
            ),
            dataset_config_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "algorithm/configs/dataset_config.yaml"
            ),
            iters=10,
            lr=1e-4,
            batch_size=1,
            optimizer="adamw",
            backbone="resnet50",
            checkpoint_after_iter=0,
            checkpoint_load_iter=0,
            temp_path="temp",
            device="cuda",
            threshold=0.7,
            num_classes=91,
            panoptic_segmentation=False,
    ):

        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(GemLearner, self).__init__(
            iters=iters,
            lr=lr,
            batch_size=batch_size,
            optimizer=optimizer,
            backbone=backbone,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            temp_path=temp_path,
            device=device,
            threshold=threshold,
        )

        # Add arguments to a structure like in the original implementation
        self.args = utils.load_config(model_config_path)
        self.datasetargs = utils.load_config(dataset_config_path)
        self.args.backbone = self.backbone
        self.args.device = self.device
        self.args.num_classes = num_classes
        self.args.dataset_file = "coco"

        if panoptic_segmentation:
            self.args.masks = True
            self.args.dataset_file = "coco_panoptic"
        else:
            self.args.masks = False
            self.args.dataset_file = "coco"

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
        self.fusion_method = 'sc_avg'

    def save(self, path, verbose=False):
        """
        This method is used to save a trained model.
        Provided with the path, absolute or relative, including a *folder* name, it creates a directory with the name
        of the *folder* provided and saves the model inside with a proper format and a .json file with metadata.
        :param path: for the model to be saved, including the folder name
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """

        if self.model is None and self.ort_session is None:
            raise UserWarning("No model is loaded, cannot save.")

        folder_name, _, tail = self.__extract_trailing(path)  # Extract trailing folder name from path
        # Also extract folder name without any extension if extension is erroneously provided
        folder_name_no_ext = folder_name.split(sep='.')[0]

        # Extract path without folder name, by removing folder name from original path
        path_no_folder_name = path.replace(folder_name, '')
        # If tail is '', then path was a/b/c/, which leaves a trailing double '/'
        if tail == '':
            path_no_folder_name = path_no_folder_name[0:-1]  # Remove one '/'

        # Create model directory
        full_path_to_model_folder = path_no_folder_name + folder_name_no_ext
        os.makedirs(full_path_to_model_folder, exist_ok=True)

        if not os.path.exists(path):
            os.makedirs(path)

        model_metadata = {"model_paths": [folder_name_no_ext + ".pth"], "framework": "pytorch", "format": "pth",
                          "has_data": False, "inference_params": {'threshold': self.threshold}, "optimized": False,
                          "optimizer_info": {}, "backbone": self.backbone}

        custom_dict = {'state_dict': self.model.state_dict()}
        torch.save(custom_dict, os.path.join(full_path_to_model_folder, model_metadata["model_paths"][0]))
        if verbose:
            print("Saved Pytorch model.")

        with open(os.path.join(full_path_to_model_folder, folder_name_no_ext + ".json"), 'w') as outfile:
            json.dump(model_metadata, outfile)

    def load(self, path, verbose=False):
        """
        Loads the model from inside the path provided, based on the metadata .json file included.
        :param path: path of the directory the model was saved
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        model_name, _, _ = self.__extract_trailing(path)  # Trailing folder name from the path provided

        if os.path.exists(os.path.join(path, model_name + ".json")):
            with open(os.path.join(path, model_name + ".json")) as metadata_file:
                metadata = json.load(metadata_file)
            self.threshold = metadata['inference_params']['threshold']
        else:
            raise UserWarning('No ' + os.path.join(path, model_name + ".json") + ' found. Please have a check')

        model_path = os.path.join(path, metadata['model_paths'][0])

        self.__create_model()
        self.model_without_ddp.load_state_dict(torch.load(model_path)['state_dict'])
        if verbose:
            print("Loaded Pytorch model.")
        return True

    def __load_checkpoint(self, path):
        """
        Internal method for loading a saved checkpoint.
        :param path: path of the saved model.
        :type path: str
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

    def fit(self, m1_train_edataset=None,
            m2_train_edataset=None,
            annotations_folder=None,
            m1_train_annotations_file=None,
            m2_train_annotations_file=None,
            m1_train_images_folder=None,
            m2_train_images_folder=None,
            out_dir='outputs',
            trial_dir='trial',
            logging_path='',
            silent=False,
            verbose=True,
            m1_val_edataset=None,
            m2_val_edataset=None,
            m1_val_annotations_file=None,
            m2_val_annotations_file=None,
            m1_val_images_folder=None,
            m2_val_images_folder=None,
            ):
        """
        This method is used for training the algorithm on a train dataset and validating on a val dataset.
        :param m1_train_edataset: Object that holds the training dataset for the first modality, defaults to 'None'
        :type m1_train_edataset: ExternalDataset class object or DatasetIterator class object, optional
        :param m2_train_edataset: Object that holds the training dataset for the second modality, defaults to 'None'
        :type m2_train_edataset: ExternalDataset class object or DatasetIterator class object, optional
        :param annotations_folder: Folder name of the annotations json file. This folder should be contained in the
        dataset path provided, defaults to 'None'
        :type annotations_folder: str, optional
        :param m1_train_annotations_file: Filename of the train annotations json file for the first modality. This file
        should be contained in the *annotations_folder*, defaults to 'None'
        :type m1_train_annotations_file: str, optional
        :param m2_train_annotations_file: Filename of the train annotations json file for the second modality. This file
        should be contained in the *annotations_folder*, defaults to 'None'
        :type m2_train_annotations_file: str, optional
        :param m1_train_images_folder: Name of the folder that contains the train dataset images for the first modality.
        This folder should be contained in the dataset path provided. Note that this is a folder name, not a path,
        defaults to 'None'
        :type m1_train_images_folder: str, optional
        :param m2_train_images_folder: Name of the folder that contains the train dataset images for the second
        modality. This folder should be contained in the dataset path provided. Note that this is a folder name, not a
        path, defaults to 'None'
        :type m2_train_images_folder: str, optional
        :param out_dir: Path to where checkpoints are saved. If the path does not exist, it will be created, defaults to
        'outputs'
        :type out_dir: str, optional
        :param trial_dir: Directory in the *out_dir* where the checkpoints will be saved, defaults to 'trial'
        :type trial_dir: str, optional
        :param logging_path: Path to where TensorBoard logs are stored. If path is not '' or None, logging is activated,
        defaults to ''
        :type logging_path: str, optional
        :param silent: If 'True', the printed output is minimal, defaults to 'False'
        :type silent: bool, optional
        :param verbose: If 'True', maximum verbosity is enabled, unless *silent* is True, defaults to 'True'
        :type verbose: bool, optional
        :param m1_val_edataset: Object that holds the validation dataset for the first modality, defaults to 'None'
        :type m1_val_edataset: ExternalDataset class object or DatasetIterator class object, optional
        :param m2_val_edataset: Object that holds the validation dataset for the second modality, defaults to 'None'
        :type m2_val_edataset: ExternalDataset class object or DatasetIterator class object, optional
        :param m1_val_annotations_file: Filename of the validation annotations json file for the first modality. This
        file should be contained in the dataset path provided in the annotations folder provided, defaults to 'None'
        :type m1_val_annotations_file: str, optional
        :param m2_val_annotations_file: Filename of the validation annotations json file for the second modality. This
        file should be contained in the dataset path provided in the annotations folder provided, defaults to 'None'
        :type m2_val_annotations_file: str, optional
        :param m1_val_images_folder: Folder name that contains the validation images for the first modality. This folder
        should be contained in the dataset path provided. Note that this is a folder name, not a path, defaults to
        'None'
        :type m1_val_images_folder: str, optional
        :param m2_val_images_folder: Folder name that contains the validation images for the second modality. This
        folder should be contained in the dataset path provided. Note that this is a folder name, not a path, defaults
        to 'None'
        :return: Returns stats regarding the last evaluation ran.
        :rtype: dict
        """
        dataset_location = os.path.join(self.datasetargs.dataset_root, self.datasetargs.dataset_name)
        if m1_train_edataset is None:
            m1_train_edataset = ExternalDataset(dataset_location, 'coco')
        if m2_train_edataset is None:
            m2_train_edataset = ExternalDataset(dataset_location, 'coco')
        if m1_val_edataset is None:
            m1_val_edataset = ExternalDataset(dataset_location, 'coco')
        if m2_val_edataset is None:
            m2_val_edataset = ExternalDataset(dataset_location, 'coco')

        if annotations_folder is None:
            annotations_folder = self.datasetargs.annotations_folder
        if m1_train_annotations_file is None:
            m1_train_annotations_file = self.datasetargs.m1_train_annotations_file
        if m2_train_annotations_file is None:
            m2_train_annotations_file = self.datasetargs.m2_train_annotations_file
        if m1_train_images_folder is None:
            m1_train_images_folder = self.datasetargs.m1_train_images_folder
        if m2_train_images_folder is None:
            m2_train_images_folder = self.datasetargs.m2_train_images_folder
        if m1_val_annotations_file is None:
            m1_val_annotations_file = self.datasetargs.m1_val_annotations_file
        if m2_val_annotations_file is None:
            m2_val_annotations_file = self.datasetargs.m2_val_annotations_file
        if m1_val_images_folder is None:
            m1_val_images_folder = self.datasetargs.m1_val_images_folder
        if m2_val_images_folder is None:
            m2_val_images_folder = self.datasetargs.m2_val_images_folder

        if silent:
            verbose = False

        train_stats = {}
        test_stats = {}
        coco_evaluator = None

        if trial_dir is not None:
            output_dir = os.path.join(out_dir, trial_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
                output_dir = os.path.join(out_dir, trial_dir + '_' + current_time)
                os.makedirs(output_dir)

        if logging_path != '' and logging_path is not None:
            logging = True
            if not os.path.exists(logging_path):
                os.mkdir(logging_path)
            writer = SummaryWriter(logging_path)
        else:
            logging = False
            writer = None

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
            checkpoint = output_dir / f'checkpoint{self.checkpoint_load_iter:04}.pth'
            self.__load_checkpoint(checkpoint)
            if not silent:
                print("Loaded" + f'checkpoint{self.checkpoint_load_iter:04}.pth')

        device = torch.device(self.device)
        m1_dataset_train = self.__prepare_dataset(
            m1_train_edataset,
            image_set="m1_train",
            images_folder_name=m1_train_images_folder,
            annotations_folder_name=annotations_folder,
            annotations_file_name=m1_train_annotations_file
        )

        train_seed = m1_dataset_train.set_seed()
        m1_sampler_train_main = RandomSampler(m1_dataset_train)
        m1_sampler_seed = m1_sampler_train_main.set_seed()

        m2_dataset_train = self.__prepare_dataset(
            m2_train_edataset,
            image_set="m2_train",
            images_folder_name=m2_train_images_folder,
            annotations_folder_name=annotations_folder,
            annotations_file_name=m2_train_annotations_file,
            seed=train_seed
        )
        m2_sampler_train_main = RandomSampler(m2_dataset_train, m1_sampler_seed)

        if m1_val_edataset is not None and m2_val_edataset is not None:
            m1_dataset_val = self.__prepare_dataset(
                m1_val_edataset,
                image_set="m1_val",
                images_folder_name=m1_val_images_folder,
                annotations_folder_name=annotations_folder,
                annotations_file_name=m1_val_annotations_file
            )
            m1_sampler_val_main = SequentialSampler(m1_dataset_val)
            val_seed = m1_dataset_val.set_seed()

            m2_dataset_val = self.__prepare_dataset(
                m2_val_edataset,
                image_set="m2_val",
                images_folder_name=m2_val_images_folder,
                annotations_folder_name=annotations_folder,
                annotations_file_name=m2_val_annotations_file,
                seed=val_seed
            )

        if not self.args.distributed:
            m1_sampler_train = m1_sampler_train_main
            m2_sampler_train = m2_sampler_train_main

            if m1_val_edataset is not None and m2_val_edataset is not None:
                m1_sampler_val = m1_sampler_val_main
        else:
            m1_sampler_train = DistributedSamplerWrapper(m1_sampler_train_main)
            m2_sampler_train = DistributedSamplerWrapper(m2_sampler_train_main)

            if m1_val_edataset is not None and m2_val_edataset is not None:
                m1_sampler_val = DistributedSamplerWrapper(m1_sampler_val_main)

        # Starting from here, code has been modified from https://github.com/facebookresearch/detr/blob/master/main.py

        m1_batch_sampler_train = torch.utils.data.BatchSampler(
            m1_sampler_train, self.batch_size, drop_last=True)

        m2_batch_sampler_train = torch.utils.data.BatchSampler(
            m2_sampler_train, self.batch_size, drop_last=True)

        m1_data_loader_train = DataLoader(
            m1_dataset_train,
            batch_sampler=m1_batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers
        )

        m2_data_loader_train = DataLoader(
            m2_dataset_train,
            batch_sampler=m2_batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers
        )

        if m1_val_edataset is not None and m2_val_edataset is not None:
            m1_data_loader_val = DataLoader(
                m1_dataset_val,
                self.batch_size,
                sampler=m1_sampler_val,
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=self.args.num_workers
            )
            base_ds = get_coco_api_from_dataset(m1_dataset_val)

        if not silent:
            print("Start training")
        start_time = time.time()
        for self.epoch in range(self.checkpoint_load_iter, self.iters):
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                m1_data_loader_train,
                m2_data_loader_train,
                self.torch_optimizer,
                device,
                self.epoch,
                self.args.clip_max_norm,
                # verbose=verbose,
                silent=silent
            )
            self.lr_scheduler.step()
            checkpoint_paths = [os.path.join(output_dir, 'checkpoint.pth')]
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
            if m1_val_edataset is not None:
                test_stats, coco_evaluator = evaluate(
                    self.model, self.criterion, self.postprocessors,
                    m1_data_loader_val, m1_data_loader_val, base_ds,
                    device, self.temp_path,
                    verbose=verbose, silent=silent)

            if logging:
                for k, v in train_stats.items():
                    if isinstance(v, list):
                        v = v[0]
                    writer.add_scalar(f'train_{k}', v, self.epoch + 1)
                if m1_val_edataset is not None:
                    for k, v in test_stats.items():
                        if isinstance(v, list):
                            v = v[0]
                        writer.add_scalar(f'test_{k}', v, self.epoch + 1)

            new_train_seed = m1_dataset_train.set_seed()
            m2_dataset_train.set_seed(new_train_seed)

            if m1_val_edataset is not None and m2_val_edataset is not None:
                new_val_seed = m1_dataset_val.set_seed()
                m2_dataset_val.set_seed(new_val_seed)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        if logging:
            writer.close()

        if not silent:
            print('Training time {}'.format(total_time_str))
        if m1_val_edataset is not None:
            return {"train_stats": train_stats, "test_stats": test_stats}
        return train_stats

    def eval(self,
             m1_edataset=None,
             m2_edataset=None,
             m1_images_folder='m1_val2017',
             m2_images_folder='m2_val2017',
             annotations_folder='Annotations',
             m1_annotations_file='m1_instances_val2017.json',
             m2_annotations_file='m2_instances_val2017.json',
             verbose=True,
             ):
        """
        This method is used to evaluate a trained model on an evaluation dataset.
        :param m1_edataset: Object that holds the evaluation dataset for the first modality, defaults to 'None'
        :type m1_edataset: ExternalDataset class object or DatasetIterator class object, optional
        :param m2_edataset: Object that holds the evaluation dataset for the second modality, defaults to 'None'
        :type m2_edataset: ExternalDataset class object or DatasetIterator class object, optional
        :param m1_images_folder: Folder name that contains the dataset images for the first modality. This folder should
        be contained in the dataset path provided. Note that this is a folder name, not a path, defaults to 'm1_val2017'
        :type m1_images_folder: str, optional
        :param m2_images_folder: Folder name that contains the dataset images for the second modality. This folder
        should be contained in the dataset path provided. Note that this is a folder name, not a path, defaults to
        'm2_val2017'
        :type m2_images_folder: str, optional
        :param annotations_folder: Folder name of the annotations json file. This file should be contained in the
        dataset path provided, defaults to 'Annotations'
        :type annotations_folder: str, optional
        :param m1_annotations_file: Filename of the annotations json file for the first modality. This file should be
        contained in the dataset path provided, defaults to 'm1_instances_val2017'
        :type m1_annotations_file: str, optional
        :param m2_annotations_file: Filename of the annotations json file for the second modality. This file should be
        contained in the dataset path provided, defaults to 'm2_instances_val2017
        :type m2_annotations_file: str, optional
        :param verbose: If 'True', maximum verbosity is enabled, defaults to 'True'
        :type verbose: bool, optional
        :return: Returns stats regarding evaluation.
        :rtype: dict
        """
        if self.model is None:
            raise UserWarning('A model should be loaded first')

        dataset_location = os.path.join(self.datasetargs.dataset_root, self.datasetargs.dataset_name)
        if m1_edataset is None:
            m1_edataset = ExternalDataset(dataset_location, 'coco')
        if m2_edataset is None:
            m2_edataset = ExternalDataset(dataset_location, 'coco')

        if annotations_folder is None:
            annotations_folder = self.datasetargs.annotations_folder
        if m1_annotations_file is None:
            m1_annotations_file = self.datasetargs.m1_val_annotations_file
        if m2_annotations_file is None:
            m2_annotations_file = self.datasetargs.m2_val_annotations_file
        if m1_images_folder is None:
            m1_images_folder = self.datasetargs.m1_val_images_folder
        if m2_images_folder is None:
            m2_images_folder = self.datasetargs.m2_val_images_folder

        if self.postprocessors is None:
            self.__create_postprocessors()

        self.__create_criterion()

        device = torch.device(self.device)

        m1_dataset = self.__prepare_dataset(
            m1_edataset,
            image_set="m1_val",
            images_folder_name=m1_images_folder,
            annotations_folder_name=annotations_folder,
            annotations_file_name=m1_annotations_file
        )
        m1_sampler_main = SequentialSampler(m1_dataset)
        m1_eval_seed = m1_dataset.set_seed()

        m2_dataset = self.__prepare_dataset(
            m2_edataset,
            image_set="m2_val",
            images_folder_name=m2_images_folder,
            annotations_folder_name=annotations_folder,
            annotations_file_name=m2_annotations_file,
            seed=m1_eval_seed
        )
        m2_sampler_main = SequentialSampler(m2_dataset)

        if not self.args.distributed:
            m1_sampler = m1_sampler_main
            m2_sampler = m2_sampler_main
        else:
            m1_sampler = DistributedSamplerWrapper(m1_sampler_main)
            m2_sampler = DistributedSamplerWrapper(m2_sampler_main)

        m1_data_loader = DataLoader(
            m1_dataset,
            self.batch_size,
            sampler=m1_sampler,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers
        )
        m2_data_loader = DataLoader(
            m2_dataset,
            self.batch_size,
            sampler=m2_sampler,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers
        )

        if isinstance(m1_edataset, ExternalDataset):
            base_ds = get_coco_api_from_dataset(m1_dataset)
        else:
            base_ds = None

        test_stats, _ = evaluate(
            self.model, self.criterion, self.postprocessors,
            m1_data_loader, m2_data_loader, base_ds, device,
            self.temp_path, verbose=verbose
        )

        return test_stats

    def infer(self, m1_image, m2_image):
        """
        This method is used to perform object detection on two images with different modalities.
        :param m1_image: Image from the first modality to run inference on
        :type m1_image: engine.data.Image class object or numpy.ndarray
        :param m2_image: Image from the second modality to run inference on
        :type m2_image: engine.data.Image class object or numpy.ndarray
        :return: Bounding box list, first modality weight, second modality weight
        :rtype: engine.target.BoundingBoxList, float, float
        """
        if not (isinstance(m1_image, Image) or isinstance(m2_image, Image)):
            m1_image = Image(m1_image)
            m2_image = Image(m2_image)

        m1_img = im.fromarray(m1_image.convert("channels_last", "rgb"))
        m2_img = im.fromarray(m2_image.convert("channels_last", "rgb"))

        scores, boxes, segmentations, contrib = detect(m1_img, m2_img, self.infer_transform, self.model,
                                                       self.postprocessors, self.device, self.threshold,
                                                       self.ort_session,
                                                       )
        weight1 = contrib[0].data
        weight1 = weight1.cpu().detach().numpy()

        weight2 = contrib[1].data
        weight2 = weight2.cpu().detach().numpy()

        normed_weight1 = weight1 / (weight1 + weight2)
        normed_weight2 = weight2 / (weight1 + weight2)

        boxlist = []
        if len(segmentations) == len(scores):
            for p, (xmin, ymin, xmax, ymax), segmentation in zip(scores.tolist(), boxes.tolist(), segmentations):
                cl = np.argmax(p)
                box = CocoBoundingBox(cl, xmin, ymin, xmax - xmin, ymax - ymin, score=p[cl], segmentation=segmentation)
                boxlist.append(box)
        else:
            for p, (xmin, ymin, xmax, ymax) in zip(scores.tolist(), boxes.tolist()):
                cl = np.argmax(p)
                box = CocoBoundingBox(cl, xmin, ymin, xmax - xmin, ymax - ymin, score=p[cl])
                boxlist.append(box)
        return BoundingBoxList(boxlist), normed_weight1, normed_weight2

    def optimize(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def download(self, path=None, mode="pretrained_gem", verbose=False):
        """
        Download utility.
        :param path: Path to locally stored (pretrained) models, defaults to 'None'
        :type path: str, optional
        :param mode: Determines the files that will be downloaded. Valid values are: "weights_detr", "pretrained_detr",
        "pretrained_gem", "test_data_l515" and "test_data_sample_images". In case of "weights_detr", the weigths for
        single modal DETR with resnet50 backbone are downloaded. In case of "pretrained_detr", the weigths for single
        modal pretrained DETR with resnet50 backbone are downloaded. In case of "pretrained_gem", the weights from
        'gem_scavg_e294_mAP0983_rn50_l515_7cls.pth' (backbone: 'resnet50', fusion_method: 'scalar averaged', trained on
        RGB-Infrared l515_dataset are downloaded. In case of "test_data_l515", the RGB-Infrared l515 dataset is
        downloaded from the OpenDR server. In case of "test_data_sample images", two sample images for testing the infer
        function are downloaded, defaults to 'pretrained_gem'
        :type mode: str, optional
        :param verbose: Enables the maximum verbosity.
        :type verbose: bool, optional
        """
        valid_modes = ["weights_detr", "pretrained_detr", "pretrained_gem", "test_data_l515",
                       "test_data_sample_dataset",
                       "test_data_sample_images"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained_detr" or mode == "weights_detr":
            supported_backbones = ['resnet50', 'resnet101']
            if self.backbone in supported_backbones:
                pass
            else:
                raise UserWarning("Backbone {} currently not supported, valid backbones are: {}".format(
                    self.backbone, supported_backbones))
            self.__create_model()
            if mode == 'pretrained_detr':
                pretrained = True
            else:
                pretrained = False
            torch.hub.set_dir(self.temp_path)
            detr_model = torch.hub.load(
                'facebookresearch/detr',
                'detr_{}'.format(self.backbone),
                pretrained=pretrained,
                return_postprocessor=False
            )
            if self.args.num_classes != 91:
                detr_model.class_embed = torch.nn.Linear(
                    in_features=detr_model.class_embed.in_features,
                    out_features=self.args.num_classes + 1)

            pretrained_dict = detr_model.state_dict()
            backbone_ir_entries_dict = {k.replace('backbone', 'backbone_ir'): v for k, v in pretrained_dict.items() if
                                        'backbone' in k}
            pretrained_dict.update(backbone_ir_entries_dict)
            if verbose:
                print("Loading detr_{} weights (partially)...".format(self.backbone))
            self.model_without_ddp.load_state_dict(pretrained_dict, strict=False)
            if verbose:
                print("Weights Loaded.")

        elif mode == 'pretrained_gem':
            self.__create_model()
            supported_backbones = ['resnet50', 'mobilenetv2']
            if self.backbone not in supported_backbones:
                raise UserWarning(
                    "Backbone {} currently not supported, valid backbones are: {}".format(
                        self.backbone, supported_backbones))
            if self.backbone == 'resnet50':
                model_file = 'gem_scavg_e294_mAP0983_rn50_l515_7cls.pth'
            elif self.backbone == 'mobilenetv2':
                model_file = 'gem_scavg_e1106_mAP0833_mnet2_l515_7cls.pth'
            pretrained_model_local_path = os.path.join(
                path, "pretrained_models/{}".format(model_file))
            if not os.path.exists(pretrained_model_local_path):
                pretrained_model_url = (
                        OPENDR_SERVER_URL +
                        'perception/object_detection_2d/gem/models/{}'.format(model_file)
                )
                if not os.path.exists(os.path.join(path, 'pretrained_models')):
                    os.makedirs(os.path.join(path, 'pretrained_models'))
                # Download pretrained_model from ftp server
                if verbose:
                    print("Downloading pretrained model from " + OPENDR_SERVER_URL)
                urlretrieve(pretrained_model_url, pretrained_model_local_path)

            pretrained_model = torch.load(pretrained_model_local_path, map_location='cpu')
            pretrained_dict = pretrained_model['model']

            if verbose:
                print("Loading gem_l515 weights...")
            self.model_without_ddp.load_state_dict(pretrained_dict, strict=True)
            if verbose:
                print("Weights Loaded.")

        elif mode == "test_data_l515":
            url = OPENDR_SERVER_URL + "perception/object_detection_2d/gem/l515_dataset.zip"
            dataset_root = path
            if not os.path.exists(dataset_root):
                os.makedirs(dataset_root)
            if not os.path.exists(os.path.join(dataset_root, 'l515_dataset')):
                print("Downloading l515_dataset...")
                urlretrieve(url, os.path.join(dataset_root, 'l515_dataset.zip'))
                print("Downloaded.")
            if os.path.exists(os.path.join(dataset_root, 'l515_dataset.zip')):
                with zipfile.ZipFile(os.path.join(dataset_root, 'l515_dataset.zip'), 'r') as zip_ref:
                    zip_ref.extractall(dataset_root)
                os.remove(os.path.join(dataset_root, 'l515_dataset.zip'))

        elif mode == "test_data_sample_dataset":
            url = OPENDR_SERVER_URL + "perception/object_detection_2d/gem/sample_dataset.zip"
            dataset_root = path
            if not os.path.exists(dataset_root):
                os.makedirs(dataset_root)
            if not os.path.exists(os.path.join(dataset_root, 'sample_dataset')):
                print("Downloading sample_dataset...")
                urlretrieve(url, os.path.join(dataset_root, 'sample_dataset.zip'))
                print("Downloaded.")
            if os.path.exists(os.path.join(dataset_root, 'sample_dataset.zip')):
                with zipfile.ZipFile(os.path.join(dataset_root, 'sample_dataset.zip'), 'r') as zip_ref:
                    zip_ref.extractall(dataset_root)
                os.remove(os.path.join(dataset_root, 'sample_dataset.zip'))

        elif mode == "test_data_sample_images":
            path = os.path.join(path, 'sample_images')
            if not os.path.exists(os.path.join(path, 'rgb')):
                os.makedirs(os.path.join(path, 'rgb'))
            if not os.path.exists(os.path.join(path, 'aligned_infra')):
                os.makedirs(os.path.join(path, 'aligned_infra'))
            if not (os.path.exists(os.path.join(path, 'rgb/2021_04_22_21_35_47_852516.jpg')) and
                    os.path.exists(os.path.join(path, 'aligned_infra/2021_04_22_21_35_47_852516.jpg'))):
                img1_url = (OPENDR_SERVER_URL +
                            "perception/object_detection_2d/gem/sample_images/rgb/2021_04_22_21_35_47_852516.jpg")
                img2_url = (OPENDR_SERVER_URL +
                            "perception/object_detection_2d/gem/sample_images/aligned_infra/2021_04_22_21_35_47_852516.jpg")
                urlretrieve(img1_url, os.path.join(path, 'rgb/2021_04_22_21_35_47_852516.jpg'))
                urlretrieve(img2_url, os.path.join(path, 'aligned_infra/2021_04_22_21_35_47_852516.jpg'))

    def __create_criterion(self):
        """
        Internal model for creating the criterion.
        """
        self.criterion = build_criterion(self.args)

    def __create_postprocessors(self):
        """
        Internal model for creating the postprocessors
        """
        self.postprocessors = build_postprocessors(self.args)

    def __create_optimizer(self):
        """
        Internal model for creating the optimizer.
        """
        param_dicts = [
            {"params": [p for n, p in self.model_without_ddp.named_parameters() if
                        "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if
                           "backbone" in n and p.requires_grad],
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
            warnings.warn(
                "Unavailbale optimizer specified, using adamw instead. Possible optimizers are; adam, adamw and sgd")
            self.torch_optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def __create_scheduler(self):
        """
        Internal method for creating the scheduler.
        """
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.torch_optimizer, self.args.lr_drop)

    def __create_model(self):
        """
        Internal method for creating a model, optimizer and scheduler based on the parameters in the config file.
        """
        self.ort_session = None

        device = torch.device(self.device)
        self.model = build_model(self.args, self.fusion_method, self.backbone)
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
                          annotations_file_name="instances_train2017.json",
                          seed=None
                          ):
        """
        This internal method prepares the dataset depending on what type of dataset is provided.

        If an ExternalDataset object type is provided, the method tried to prepare the dataset based on the original
        implementation, supposing that the dataset is in the COCO format. The path provided is searched for the
        images folder and the annotations file, converts the annotations file into the internal format used if needed
        and finally the CocoTrainDataset object is returned.

        If the dataset is of the DatasetIterator format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.
        :param dataset: Dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param image_set: Specifies whether the dataset is a train or validation dataset, possible values: "train" or
        "val", defaults to 'train'
        :type image_set: str, optional
        :param images_folder_name: The name of the folder that contains the image files. The default 'train2017'
        :type images_folder_name: str, optional
        :param annotations_folder_name: The folder that contains the original annotations, defaults to 'Annotations'
        :type annotations_folder_name: str, optional
        :param annotations_file_name: The .json file that contains the original annotations, defaults to
        'instances_train2017.json'
        :type annotations_file_name: str, optional
        :param seed: Seed that is used for building the dataset, defaults to 'None'
        :type seed: int, optional
        :return: Dataset
        :rtype: ExternalDataset or DatasetIterator
        """
        if not (isinstance(dataset, ExternalDataset) or isinstance(dataset, DatasetIterator)):
            raise UserWarning("dataset must be an ExternalDataset or a DatasetIterator")
        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() not in ["coco", "coco_panoptic"]:
                raise UserWarning("dataset_type must be \"COCO\" or \"COCO_PANOPTIC\"")

            if dataset.dataset_type.lower() == "coco_panoptic":
                self.arg.dataset_file = "coco_panoptic"

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
            return build_dataset(images_folder, annotations_folder, seed,
                                 annotations_file, image_set, self.args.masks,
                                 dataset.dataset_type.lower())

        # Create Map function for converting (Image, BoundingboxList) to detr format
        map_function = map_bounding_box_list_to_coco(image_set, self.args.masks)
        return MappedDatasetIterator(dataset, map_function)

    @staticmethod
    def __extract_trailing(path):
        """
        Extracts the trailing folder name or filename from a path provided in an OS-generic way, also handling
        cases where the last trailing character is a separator. Returns the folder name and the split head and tail.
        :param path: the path to extract the trailing filename or folder name from
        :type path: str
        :return: the folder name, the head and tail of the path
        :rtype: tuple of three strings
        """
        head, tail = ntpath.split(path)
        folder_name = tail or ntpath.basename(head)  # handle both a/b/c and a/b/c/
        return folder_name, head, tail
