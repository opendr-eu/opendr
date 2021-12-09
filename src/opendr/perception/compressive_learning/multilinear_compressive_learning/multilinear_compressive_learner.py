# Copyright 1996-2020 OpenDR European Project
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

# general imports
import numpy as np
import torch
from torch.utils.data import DataLoader
import tempfile
import random
import string
import time
import os
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.datasets import DatasetIterator
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR multilinear_compressive_learning imports
from opendr.perception.compressive_learning.multilinear_compressive_learning.algorithm import (
    trainers,
    CompressiveLearner,
    get_builtin_backbones
)
from opendr.perception.compressive_learning.multilinear_compressive_learning.algorithm.data import DataWrapper
from opendr.perception.compressive_learning.multilinear_compressive_learning.algorithm.trainers import (
    get_cosine_lr_scheduler,
    get_multiplicative_lr_scheduler
)

__all__ = ['get_cosine_lr_scheduler',
           'get_multiplicative_lr_scheduler',
           'get_builtin_backbones']

# constants
PRETRAINED_COMPRESSED_SHAPE = ((20, 19, 2),
                               (28, 27, 1),
                               (14, 11, 2),
                               (18, 17, 1),
                               (9, 6, 1),
                               (6, 9, 1))


class MultilinearCompressiveLearner(Learner):
    def __init__(self,
                 input_shape,
                 compressed_shape,
                 backbone,
                 n_class,
                 pretrained_backbone='',
                 init_backbone=True,
                 lr_scheduler=get_cosine_lr_scheduler(1e-3, 1e-5),
                 optimizer='adam',
                 weight_decay=1e-4,
                 n_init_iters=100,
                 iters=300,
                 batch_size=32,
                 checkpoint_after_iter=1,
                 checkpoint_load_iter=0,
                 temp_path='',
                 device='cpu',
                 test_mode=False,
                 ):

        super(MultilinearCompressiveLearner, self).__init__(batch_size=batch_size,
                                                            checkpoint_after_iter=checkpoint_after_iter,
                                                            checkpoint_load_iter=checkpoint_load_iter,
                                                            temp_path=temp_path,
                                                            device=device)

        assert pretrained_backbone in ['', 'with_classifier', 'without_classifier'],\
            'Option "pretrained_backbone" must be empty, "with_classifier" or "without_classifier". ' +\
            'Provided value: {}'.format(pretrained_backbone)

        if pretrained_backbone == 'with_classifier':
            if isinstance(backbone, str) and backbone.startswith('cifar'):
                assert n_class in [10, 100],\
                    'For `backbone`="{}" and `pretrained_backbone`="with_classifier", '.format(backbone) +\
                    '`n_class` must be 10 or 100\n' +\
                    'Provided `n_class`={}'.format(n_class)
            elif isinstance(backbone, str) and backbone.startswith('imagenet'):
                assert n_class == 1000,\
                    'For `backbone`="{}" and `pretrained_backbone`="with_classifier", '.format(backbone) +\
                    '`n_class` must be 1000\n' +\
                    'Provided `n_class`={}'.format(n_class)

        assert checkpoint_load_iter in [-1, 0],\
            'check_point_load_iter must be -1 or 0, with 0 indicating training from scratch,' +\
            '-1 indicating training from latest checkpoint if temp_path is given'

        assert optimizer in ['adam', 'sgd'],\
            'given optimizer "{}" is not supported, please select set optimizer to "adam" or "sgd"'.format(optimizer)

        self.model = CompressiveLearner(input_shape, compressed_shape, n_class, backbone, pretrained_backbone)
        self.input_shape = input_shape
        self.compressed_shape = compressed_shape
        self.backbone_classifier = backbone
        self.init_backbone = init_backbone
        self.n_class = n_class
        self.lr_scheduler = lr_scheduler
        self.n_init_epoch = n_init_iters
        self.n_epoch = iters
        self.batch_size = batch_size
        self.checkpoint_freq = checkpoint_after_iter
        self.epoch_idx = checkpoint_load_iter
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.device = device
        self.test_mode = test_mode
        self.temp_path = temp_path

    def _prepare_temp_dir(self,):
        if self.temp_path == '':
            # if temp dir not provided, create one under default system temp dir
            random_name = ''.join([random.choice(string.ascii_lowercase) for _ in range(128)]) + str(time.time())
            self.temp_dir_obj = tempfile.TemporaryDirectory(prefix=random_name)
            temp_dir = self.temp_dir_obj.name
        else:
            if not os.path.exists(self.temp_path):
                os.makedirs(self.temp_path)
            self.temp_dir_obj = None
            temp_dir = self.temp_path

        self.classifier_temp_dir = os.path.join(temp_dir, 'classifier')
        self.initializer_temp_dir = os.path.join(temp_dir, 'initializer')
        self.model_temp_dir = os.path.join(temp_dir, 'main_model')

        if not os.path.exists(self.classifier_temp_dir):
            os.mkdir(self.classifier_temp_dir)

        if not os.path.exists(self.initializer_temp_dir):
            os.mkdir(self.initializer_temp_dir)

        if not os.path.exists(self.model_temp_dir):
            os.mkdir(self.model_temp_dir)

    def _validate_dataset(self, dataset):
        """
        This internal function is used to perform basic validation of the data dimensions
        """

        if dataset is None:
            return

        if not isinstance(dataset, DatasetIterator):
            msg = 'Dataset must be an instance of `engine.datasets.DatasetIterator` class\n' +\
                  'Received an instance of type: {}'.format(type(dataset))
            raise TypeError(msg)

        x, y = dataset.__getitem__(0)

        if not isinstance(x, Image):
            msg = 'The 1st element returned by __getitem__ must be an instance of `engine.data.Image` class\n' +\
                  'Received an instance of type: {}'.format(type(x))
            raise TypeError(msg)

        if not isinstance(y, Category):
            msg = 'The 2nd element returned by __getitem__ must be an instance of `engine.target.Cateogry` class\n' +\
                  'Received an instance of type: {}'.format(type(y))
            raise TypeError(msg)

        x = x.convert("channels_last", "rgb")
        if not x.shape == tuple(self.input_shape):
            msg = 'Given input_shape does not match dimensions of data produced by dataset\n' +\
                  'input_shape: {}, data shape: {}'.format(tuple(self.input_shape), x.shape)
            raise ValueError(msg)

    def fit(self, train_set, val_set=None, test_set=None, logging_path='', silent=False, verbose=True):
        """
        Method to train the multilinear compressive learning model

        :param train_set: object that holds training samples
        :type train_set: engine.datasets.DatasetIterator
                         with __getitem__ producing (engine.data.Image, engine.target.Category)
        :param val_set: object that holds the validation samples, default to None
                        if available, `val_set` is used to select the best checkpoint for final model
        :type val_set: engine.datasets.DatasetIterator
                       with __getitem__ producing (engine.data.Image, engine.target.Category)
        :param test_set: object that holds the test samples, default to None
        :type test_set: engine.datasets.DatasetIterator
                        with __getitem__ producing (engine.data.Image, engine.target.Category)
        :param logging_path: path to save tensoboard data, default to ""
        :type logging_path: string
        :param silent: disable performance printing, default to False
        :type silent: bool
        :param verbose: enable the progress bar of each epoch, default to True
        :type verbose: bool

        :return: returns the performance curves when training
                 backbone classifier, sensing and synthesis components and compressive learning model
        :rtype: list(dict)
        """

        self._validate_dataset(train_set)
        self._validate_dataset(val_set)
        self._validate_dataset(test_set)
        self._prepare_temp_dir()

        train_loader = DataLoader(DataWrapper(train_set),
                                  batch_size=self.batch_size,
                                  pin_memory=self.device == 'cuda',
                                  shuffle=True)

        if val_set is None:
            val_loader = None
        else:
            val_loader = DataLoader(DataWrapper(val_set),
                                    batch_size=self.batch_size,
                                    pin_memory=self.device == 'cuda',
                                    shuffle=False)
        if test_set is None:
            test_loader = None
        else:
            test_loader = DataLoader(DataWrapper(test_set),
                                     batch_size=self.batch_size,
                                     pin_memory=self.device == 'cuda',
                                     shuffle=False)

        if self.test_mode and not silent:
            print('\nWARNING: training model under test mode\n')

        if logging_path != '':
            if not os.path.exists(logging_path):
                os.makedirs(logging_path)

            tensorboard_logger = SummaryWriter(logging_path)
        else:
            tensorboard_logger = None

        # train the backbone classifier if init_backbone is True
        if self.init_backbone:
            backbone_performance = self._fit_backbone(train_loader,
                                                      val_loader,
                                                      test_loader,
                                                      tensorboard_logger,
                                                      silent,
                                                      verbose)
        else:
            backbone_performance = {}

        # train the sensing and feature synthesis modules
        init_performance = self._fit_sensing_and_synthesis(train_loader,
                                                           val_loader,
                                                           test_loader,
                                                           tensorboard_logger,
                                                           silent,
                                                           verbose)

        # train the whole model
        compressive_learning_performance = self._fit_system(train_loader,
                                                            val_loader,
                                                            test_loader,
                                                            tensorboard_logger,
                                                            silent,
                                                            verbose)

        if tensorboard_logger is not None:
            tensorboard_logger.close()

        if self.temp_dir_obj is not None:
            self.temp_dir_obj.cleanup()

        performance = {'backbone_performance': backbone_performance,
                       'initialization_performance': init_performance,
                       'compressive_learning_performance': compressive_learning_performance}

        return performance

    def eval(self, dataset, silent=False, verbose=True):
        """
        This method is used to evaluate the performance of a given set of data

        :param dataset: object that holds the set of samples to evaluate
        :type dataset: engine.datasets.DatasetIterator
                       with __getitem__ producing (engine.data.Image, engine.target.Category)

        :return: a dictionary that contains `cross_entropy` and `acc` as keys
        :rtype: dict
        """

        self._validate_dataset(dataset)
        loader = DataLoader(DataWrapper(dataset),
                            batch_size=self.batch_size,
                            pin_memory=self.device == 'cuda',
                            shuffle=False)

        device = torch.device(self.device)

        self.model.to(device)
        self.model.eval()

        L = torch.nn.CrossEntropyLoss()
        n_correct = 0
        n_sample = 0
        loss = 0

        if verbose:
            loader = tqdm(loader, desc='#Evaluating ', ncols=80, ascii=True)

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long().flatten()

                predictions = self.model(inputs)
                n_sample += inputs.size(0)
                loss += L(predictions, targets).item()
                n_correct += (predictions.argmax(dim=-1) == targets).sum().item()

        metrics = {'cross_entropy': loss / n_sample,
                   'acc': n_correct / n_sample}

        if not silent:
            print('cross_entropy: {:.6f}, acc: {:.4f}'.format(metrics['cross_entropy'], metrics['acc']))

        return metrics

    def infer(self, img):
        """
        This method is used to generate class prediction given an image

        :param img: image to generate class prediction
        :type img: engine.data.Image

        :return: predicted label
        :rtype: engine.target.Category

        """

        if not isinstance(img, Image):
            msg = 'Input to `infer()` must be an instance of engine.data.Image\n' +\
                  'Received an instance of type: {}'.format(type(img))
            raise TypeError(msg)

        self.model.to(torch.device(self.device))
        self.model.eval()

        img = img.convert("channels_last", "rgb")

        if not tuple(img.shape) == tuple(self.input_shape):
            msg = 'Dimensions of the given input "{}"'.format(img.shape) +\
                  'do not match input dimensions "{}" of the model'.format(tuple(self.input_shape))
            raise ValueError(msg)

        img = np.expand_dims(img.transpose(2, 0, 1), 0)

        with torch.no_grad():
            tensor_img = torch.tensor(img, device=torch.device(self.device)).float()
            prob_prediction = torch.nn.functional.softmax(self.model(tensor_img).flatten(), dim=0)
            class_prediction = prob_prediction.argmax(dim=-1).cpu().item()
            prediction = Category(class_prediction, confidence=prob_prediction[class_prediction].cpu().item())

        return prediction

    def get_sensing_parameters(self):
        """
        This method is used to get the sensing parameters

        :return: parameters of sensing operators
        :rtype: list of numpy array

        """

        params = [x.detach().cpu().numpy() for x in list(self.model.sense_synth_module.synthesis_module.parameters())]
        # reverse the order from pytorch order to normal image order
        if len(params) == 3:
            params = [params[1], params[2], params[0]]

        return params

    def infer_from_compressed_measurement(self, measurement):
        """
        This method is used to generate class prediction given the compressed measurement

        :param measurement: compressed measurement to generate class prediction
        :type measurement: engine.data.Image

        :return: predicted label
        :rtype: engine.target.Category

        """

        if not isinstance(measurement, Image):
            msg = 'Input to `infer_from_compressed_measurement()` must be an instance of engine.data.Image\n' +\
                  'Received an instance of type: {}'.format(type(measurement))
            raise TypeError(msg)

        self.model.to(torch.device(self.device))
        self.model.eval()

        measurement = measurement.convert("channels_last")

        if not tuple(measurement.shape) == tuple(self.compressed_shape):
            msg = 'Dimensions of the given compressed measurement "{}"'.format(measurement.shape) +\
                  'do not match `compressed_shape` "{}" of the model'.format(tuple(self.compressed_shape))
            raise ValueError(msg)

        measurement = np.expand_dims(measurement.transpose(2, 0, 1), 0)

        with torch.no_grad():
            tensor_measurement = torch.tensor(measurement, device=torch.device(self.device)).float()
            prob_prediction = self.model.infer_from_measurement(tensor_measurement).flatten()
            prob_prediction = torch.nn.functional.softmax(prob_prediction, dim=0)
            class_prediction = prob_prediction.argmax(dim=-1).cpu().item()
            prediction = Category(class_prediction, confidence=prob_prediction[class_prediction].cpu().item())

        return prediction

    def save(self, path, verbose=True):
        """
        This function is used to save the current model given a directory path. Metadata and model weights
        are saved under path, i.e., `path/metadata.json` and path/model_weights.pt` will be created

        :param path: path to the directory that the model will be saved
        :type path: str
        :param verbose: whether to print acknowledge message when saving is successful, default to True
        :type verbose: bool

        """

        if not os.path.exists(path):
            os.makedirs(path)

        if isinstance(self.backbone_classifier, str):
            backbone = self.backbone_classifier
        else:
            backbone = type(self.backbone_classifier).__name__

        model_weight_file = os.path.join(path, 'model_weights.pt')
        metadata_file = os.path.join(path, 'metadata.json')

        metadata = {'framework': 'pytorch',
                    'model_paths': model_weight_file,
                    'format': 'pt',
                    'backbone': backbone,
                    'input_shape': self.input_shape,
                    'compressed_shape': self.compressed_shape,
                    'has_data': False,
                    'inference_params': {},
                    'optimized': False,
                    'optimimizer_info': {}
                    }

        try:
            torch.save(self.model.cpu().state_dict(), model_weight_file)
            if verbose:
                print('Model weights saved to {}'.format(model_weight_file))
        except Exception as error:
            raise error

        try:
            fid = open(metadata_file, 'w')
            json.dump(metadata, fid)
            fid.close()

            if verbose:
                print('Model metadata saved to {}'.format(metadata_file))

        except Exception as error:
            raise error

    def load(self, path, verbose=True):
        """
        This function is used to load a pretrained model that has been saved with .save(), given the path
        to the directory. `path/metadata.json` and `path/model_weights.pt` should exist

        :param path: path to the saved location
        :type path: str
        :param verbose: whether to print acknowledge message when loading is successful, defaul to True
        :type verbose: bool

        """

        if not os.path.exists(path):
            raise FileNotFoundError('Directory "{}" does not exist'.format(path))

        if not os.path.isdir(path):
            raise ValueError('Given path "{}" is not a directory'.format(path))

        metadata_file = os.path.join(path, 'metadata.json')
        model_weight_file = os.path.join(path, 'model_weights.pt')

        assert os.path.exists(metadata_file),\
            'Metadata file ("metadata.json") does not exist under the given path "{}"'.format(path)

        assert os.path.exists(model_weight_file),\
            'Model weights ("model_weights.pt") does not exist under the given path "{}"'.format(path)

        fid = open(metadata_file, 'r')
        metadata = json.load(fid)
        fid.close()

        assert tuple(metadata['input_shape']) == tuple(self.input_shape),\
            'Input dimensions "{}" of saved model '.format(metadata['input_shape']) +\
            'do not match input dimensions "{}" of current model instance'.format(self.input_shape)

        assert tuple(metadata['compressed_shape']) == tuple(self.compressed_shape),\
            'Compressed dimensions "{}" of saved model '.format(metadata['compressed_shape']) +\
            'do not match comressed dimensions "{}" of current model instance'.format(self.compressed_shape)

        self.model.cpu()
        self.model.load_state_dict(torch.load(model_weight_file, map_location=torch.device('cpu')))

        if verbose:
            print('Pretrained model is loaded successfully')

    def _fit_backbone(self, train_loader, val_loader, test_loader, tensorboard_logger, silent, verbose):
        """
        This is the internal function used to train the backbone classifier
        """

        if not silent:
            print('\n======= Training backbone classifier =======\n')
        trainer = trainers.ClassifierTrainer(n_epoch=self.n_epoch,
                                             epoch_idx=self.epoch_idx,
                                             lr_scheduler=self.lr_scheduler,
                                             optimizer=self.optimizer,
                                             weight_decay=self.weight_decay,
                                             temp_dir=self.classifier_temp_dir,
                                             checkpoint_freq=self.checkpoint_freq,
                                             print_freq=int(not silent),
                                             use_progress_bar=verbose,
                                             test_mode=self.test_mode)

        device = torch.device(self.device)
        backbone_performance = trainer.fit(self.model.backbone,
                                           train_loader,
                                           val_loader,
                                           test_loader,
                                           device,
                                           tensorboard_logger,
                                           logger_prefix='backbone')

        return backbone_performance

    def _fit_sensing_and_synthesis(self, train_loader, val_loader, test_loader, tensorboard_logger, silent, verbose):
        """
        This is the internal function used to train the sensing and feature synthesis components
        """

        if not silent:
            print('\n======= Training the sensing and feature synthesis modules =======\n')
        trainer = trainers.AutoRegressionTrainer(n_epoch=self.n_init_epoch,
                                                 epoch_idx=self.epoch_idx,
                                                 lr_scheduler=self.lr_scheduler,
                                                 optimizer=self.optimizer,
                                                 weight_decay=self.weight_decay,
                                                 temp_dir=self.initializer_temp_dir,
                                                 checkpoint_freq=self.checkpoint_freq,
                                                 print_freq=int(not silent),
                                                 use_progress_bar=verbose,
                                                 test_mode=self.test_mode)

        device = torch.device(self.device)
        init_performance = trainer.fit(self.model.sense_synth_module,
                                       train_loader,
                                       val_loader,
                                       test_loader,
                                       device,
                                       tensorboard_logger,
                                       logger_prefix='sensing_and_synthesis')

        return init_performance

    def _fit_system(self, train_loader, val_loader, test_loader, tensorboard_logger, silent, verbose):
        """
        This is the internal function used to train the all components of the multilinear compressive
        learning model
        """

        if not silent:
            print('\n======= Training all modules =======\n')
        trainer = trainers.ClassifierTrainer(n_epoch=self.n_epoch,
                                             epoch_idx=self.epoch_idx,
                                             lr_scheduler=self.lr_scheduler,
                                             optimizer=self.optimizer,
                                             weight_decay=self.weight_decay,
                                             temp_dir=self.model_temp_dir,
                                             checkpoint_freq=self.checkpoint_freq,
                                             print_freq=int(not silent),
                                             use_progress_bar=verbose,
                                             test_mode=self.test_mode)

        device = torch.device(self.device)
        backbone_performance = trainer.fit(self.model,
                                           train_loader,
                                           val_loader,
                                           test_loader,
                                           device,
                                           tensorboard_logger,
                                           logger_prefix='main_model')

        return backbone_performance

    def download(self, path):
        """
        This function is used to download a pretrained model given the current model specification
        Calling load(path) after this function will load the downloaded model weights

        :param path: path to the saved location. Under this path `model_weights.pt` and `metadata.json`
                     will be downloaded so different paths for different models should be given to avoid
                     overwriting previously downloaded model
        :type path: str
        """

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if isinstance(self.backbone_classifier, str) and self.backbone_classifier == 'cifar_allcnn':
            assert self.n_class in [10, 100],\
                'Only support pretrained CIFAR10 or CIFAR100 model.\n' +\
                'Current model specification has {} classes'.format(self.n_class)

            compressed_shape = tuple(self.compressed_shape)
            assert compressed_shape in PRETRAINED_COMPRESSED_SHAPE,\
                'Current `compressed_shape` does not match supported shapes. Supported shapes includes\n' +\
                '\n'.join(PRETRAINED_COMPRESSED_SHAPE)

            server_url = os.path.join(OPENDR_SERVER_URL,
                                      'perception',
                                      'compressive_learning',
                                      'multilinear_compressive_learning')

            dataset = 'cifar10' if self.n_class == 10 else 'cifar100'
            backbone = 'allcnn'
            model_name = '{}_{}_{}_{}_{}'.format(dataset,
                                                 backbone,
                                                 self.compressed_shape[0],
                                                 self.compressed_shape[1],
                                                 self.compressed_shape[2])

            metadata_url = os.path.join(server_url, '{}.json'.format(model_name))
            metadata_file = os.path.join(path, 'metadata.json')
            urlretrieve(metadata_url, metadata_file)

            weights_url = os.path.join(server_url, '{}.pt'.format(model_name))
            weights_file = os.path.join(path, 'model_weights.pt')
            urlretrieve(weights_url, weights_file)
            print('Pretrained model downloaded to the following directory\n{}'.format(path))
        else:
            raise UserWarning('Only pretrained model for built-in backbone can be downloaded')

    def optimize(self):
        pass

    def reset(self):
        pass
