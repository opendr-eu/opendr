# Copyright 2020 Aristotle University of Thessaloniki
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
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader

from engine.learners import Learner
from engine.datasets import ExternalDataset, DatasetIterator

from datasets.coco import build_coco_dataset

from utils import collate_fn
from model import DETRmodel
from detr_engine import train_one_epoch, evaluate
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

class ObjectDetection2DLearner(Learner):
    # 1. The default values in constructor arguments can be set according to the algorithm.
    # 2. Some of the shared parameters, e.g. lr_schedule, backbone, etc., can be skipped here if not needed
    #    by the implementation.
    # 3. TODO Make sure the naming of the arguments is the same as the parent class arguments to keep it consistent
    #     for the end user.
    def __init__(self, lr=0.001, iters=10, batch_size=64, optimizer='adamw',
                 lr_schedule='', backbone='resnet50', network_head='',
                 checkpoint_after_iter=0, checkpoint_load_iter=0,
                 temp_path='temp', device='cuda', threshold=0.0,
                 scale=1.0, weight_decay=0.0001, lr_drop=200,
                 num_classes=91, num_queries=100, num_workers=2,
                 loss_ce=1, loss_bbox=5, loss_giou=2, eos_coef=0.1,
                 pretrained=True,  experiment_name='default'):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(ObjectDetection2DLearner, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer,
                                             lr_schedule=lr_schedule, backbone=backbone, network_head=network_head,
                                             checkpoint_after_iter=checkpoint_after_iter,
                                             checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                             device=device, threshold=threshold, scale=scale)


        supportedOptimizers = ['adam', 'adamw', 'sgd']
        if self.optimizer.lower() not in supportedOptimizers:
            raise ValueError(self.optimizer + " not a valid optimizer. Supported optimizers:" + str(supportedOptimizers))

        supportedBackbones = ["resnet50", "resnet101"]
        if self.backbone.lower() not in supportedBackbones:
            raise ValueError(self.backbone + " not a valid backbone. Supported backbones:" + str(supportedBackbones))

        # TODO Make sure to do appropriate typechecks and provide valid default values for all custom parameters used.
        self.weight_decay = weight_decay
        self.lr_drop = lr_drop
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.pretrained = pretrained

        # Loss parameters
        self.loss_ce = loss_ce
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.eos_coef = eos_coef

    def init_model(self):
        self.model = DETRmodel(self.num_classes, self.num_queries, self.backbone, self.pretrained)

    # All methods below are dummy implementations of the abstract methods that are inherited.
    def save(self, path):
        pass

    def load(self, path):
        pass

    def optimize(self, params):
        pass

    def reset(self):
        pass

    def fit(self, dataset, val_dataset=None,
            annotations_folder='Annotations',
            train_images_folder='train2017',
            train_annotations_file='instances_train2017.json',
            val_images_folder='val2017',
            val_annotations_file='instances_val2017.json',
            logging_path='', silent=False, verbose=False):

        device = torch.device(self.device)

        dataset_train = self.__prepare_dataset(dataset,
                                      image_set="train",
                                      images_folder_name=train_images_folder,
                                      annotations_folder_name=annotations_folder,
                                      annotations_file_name=train_annotations_file)

        data_loader_train = DataLoader(dataset_train, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)

        # Model initialization
        if self.model is None:
            self.init_model()
            self.model.to(device)

        if self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.lr_drop)

        matcher = HungarianMatcher()

        weight_dict = {'loss_ce': self.loss_ce, 'loss_bbox': self.loss_bbox, 'loss_giou': self.loss_giou}

        losses = ['labels', 'boxes', 'cardinality']

        criterion = SetCriterion(self.num_classes-1, matcher, weight_dict,
                                 eos_coef=self.eos_coef, losses=losses)
        criterion = criterion.to(device)

        for epoch in range(self.iters):
            train_loss = train_one_epoch(data_loader_train, self.model,
                                         criterion, optimizer, device,
                                         scheduler=scheduler,epoch=epoch)

    def eval(self, dataset):
        pass

    def infer(self, batch, tracked_bounding_boxes=None):
        # In this infer dummy implementation, a custom argument is added as optional, so as not to change the basic
        # signature of the abstract method.
        # TODO The implementation must make sure it throws an appropriate error if the custom argument is needed and
        #  not provided (None).
        pass

    @staticmethod
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
            if dataset.dataset_type.lower() != "coco":
                raise UserWarning("dataset_type must be \"COCO\"")

            # Get images folder
            images_folder = os.path.join(dataset.path, images_folder_name)
            if not(os.path.isdir(images_folder)):
                raise UserWarning("Didn't find \"" + images_folder_name +
                                  "\" folder in the dataset path provided.")

            # Get annotations file
            if annotations_folder_name == "":
                annotations_file = os.path.join(dataset.path, annotations_file_name)
            else:
                annotations_file_dir = os.path.join(dataset.path, annotations_folder_name)
                annotations_file = os.path.join(annotations_file_dir, annotations_file_name)
            if not(os.path.isfile(annotations_file)):
                raise UserWarning("Didn't find \"" + annotations_file +
                                  "\" file in the dataset path provided.")

            return build_coco_dataset(images_folder, annotations_file, image_set, self.return_masks)
        elif isinstance(dataset, DatasetIterator):
            return dataset
