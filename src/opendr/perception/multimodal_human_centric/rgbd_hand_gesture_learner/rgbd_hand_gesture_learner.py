# Copyright 2020-2021 OpenDR European Project
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
import warnings

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.datasets import DatasetIterator
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR imports
from opendr.perception.multimodal_human_centric.rgbd_hand_gesture_learner.algorithm import (
    architectures,
    get_builtin_architectures,
    get_pretrained_architectures,
    RgbdDataset,
    DataWrapper,
    get_hand_gesture_dataset
)
from opendr.perception.compressive_learning.multilinear_compressive_learning.algorithm.trainers import (
    get_cosine_lr_scheduler,
    get_multiplicative_lr_scheduler,
    ClassifierTrainer
)

# constants
PRETRAINED_MODEL = ['mobilenet_v2']

__all__ = ['architectures',
           'get_builtin_architectures',
           'get_pretrained_architectures',
           'get_hand_gesture_dataset',
           'get_cosine_lr_scheduler',
           'get_multiplicative_lr_scheduler']


class RgbdHandGestureLearner(Learner):
    def __init__(self,
                 n_class,
                 architecture,
                 pretrained=False,
                 lr_scheduler=get_cosine_lr_scheduler(1e-3, 1e-5),
                 optimizer='adam',
                 weight_decay=1e-4,
                 iters=200,
                 batch_size=32,
                 n_workers=4,
                 checkpoint_after_iter=1,
                 checkpoint_load_iter=0,
                 temp_path='',
                 device='cpu',
                 test_mode=False,
                 ):
        super(RgbdHandGestureLearner, self).__init__(batch_size=batch_size,
                                                     checkpoint_after_iter=checkpoint_after_iter,
                                                     checkpoint_load_iter=checkpoint_load_iter,
                                                     temp_path=temp_path,
                                                     device=device)

        if isinstance(architecture, str):
            assert architecture in get_builtin_architectures(),\
                'The given architecture "{}" is not implemented\n'.format(architecture) +\
                'Supported architectures include:\n' +\
                '\n'.join(get_builtin_architectures())

            if pretrained and architecture not in get_pretrained_architectures():
                warnings.warn('There is no pretrained model for architecture "{}"'.format(architecture))

            self.model = architectures[architecture](num_classes=n_class, pretrained=pretrained)
        else:
            self.model = architecture

        assert checkpoint_load_iter < iters,\
            '`checkpoint_load_iter` must be less than `iters`\n' +\
            'Received checkpoint_load_iter={}, iters={}'.format(checkpoint_load_iter, iters)

        assert optimizer in ['adam', 'sgd'],\
            'given optimizer "{}" is not supported, please select set optimizer to "adam" or "sgd"'.format(optimizer)

        self.n_class = n_class
        self.architecture = architecture
        self.lr_scheduler = lr_scheduler
        self.n_epoch = iters
        self.batch_size = batch_size
        self.n_workers = n_workers
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
            self.temp_dir = self.temp_dir_obj.name
        else:
            if not os.path.exists(self.temp_path):
                os.makedirs(self.temp_path)
            self.temp_dir_obj = None
            self.temp_dir = self.temp_path

    def _validate_x(self, x):
        if not isinstance(x, Image):
            msg = 'The 1st element returned by __getitem__ must be an instance of `engine.data.Image` class\n' +\
                  'Received an instance of type: {}'.format(type(x))
            raise TypeError(msg)

        x = x.convert("channels_first")
        if x.shape[0] != 4:
            msg = 'The first dimension of data produced by dataset must be 4\n' +\
                  'Received input of shape: {}'.format(x.shape)
            raise ValueError(msg)

    def _validate_y(self, y):
        if not isinstance(y, Category):
            msg = 'The 2nd element returned by __getitem__ must be an instance of `engine.target.Cateogry` class\n' +\
                  'Received an instance of type: {}'.format(type(y))
            raise TypeError(msg)

    def _validate_dataset(self, dataset):
        """
        This internal function is used to perform basic validation of the data dimensions
        """

        if dataset is None:
            return

        if not isinstance(dataset, RgbdDataset):
            if not isinstance(dataset, DatasetIterator):
                msg = 'Dataset must be an instance of `engine.datasets.DatasetIterator` class\n' +\
                      'Received an instance of type: {}'.format(type(dataset))
                raise TypeError(msg)
            else:
                x, y = dataset.__getitem__(0)
                self._validate_x(x)
                self._validate_y(y)

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

        :return: the performance curves
        :rtype: list(dict)
        """

        self._validate_dataset(train_set)
        self._validate_dataset(val_set)
        self._validate_dataset(test_set)
        self._prepare_temp_dir()

        if isinstance(train_set, RgbdDataset):
            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      pin_memory=self.device == 'cuda',
                                      num_workers=self.n_workers,
                                      shuffle=True)
        else:
            train_loader = DataLoader(DataWrapper(train_set),
                                      batch_size=self.batch_size,
                                      num_workers=self.n_workers,
                                      pin_memory=self.device == 'cuda',
                                      shuffle=True)

        if val_set is None:
            val_loader = None
        else:
            if isinstance(val_set, RgbdDataset):
                val_loader = DataLoader(val_set,
                                        batch_size=self.batch_size,
                                        num_workers=self.n_workers,
                                        pin_memory=self.device == 'cuda',
                                        shuffle=False)
            else:
                val_loader = DataLoader(DataWrapper(val_set),
                                        batch_size=self.batch_size,
                                        num_workers=self.n_workers,
                                        pin_memory=self.device == 'cuda',
                                        shuffle=False)

        if test_set is None:
            test_loader = None
        else:
            if isinstance(test_set, RgbdDataset):
                test_loader = DataLoader(test_set,
                                         batch_size=self.batch_size,
                                         num_workers=self.n_workers,
                                         pin_memory=self.device == 'cuda',
                                         shuffle=False)
            else:
                test_loader = DataLoader(DataWrapper(test_set),
                                         batch_size=self.batch_size,
                                         num_workers=self.n_workers,
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

        trainer = ClassifierTrainer(n_epoch=self.n_epoch,
                                    epoch_idx=self.epoch_idx,
                                    lr_scheduler=self.lr_scheduler,
                                    optimizer=self.optimizer,
                                    weight_decay=self.weight_decay,
                                    temp_dir=self.temp_dir,
                                    checkpoint_freq=self.checkpoint_freq,
                                    print_freq=int(not silent),
                                    use_progress_bar=verbose,
                                    test_mode=self.test_mode)

        device = torch.device(self.device)
        performance = trainer.fit(self.model,
                                  train_loader,
                                  val_loader,
                                  test_loader,
                                  device,
                                  tensorboard_logger,
                                  logger_prefix='performance')

        if tensorboard_logger is not None:
            tensorboard_logger.close()

        if self.temp_dir_obj is not None:
            self.temp_dir_obj.cleanup()

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
        if isinstance(dataset, RgbdDataset):
            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.n_workers,
                                pin_memory=self.device == 'cuda',
                                shuffle=False)
        else:
            loader = DataLoader(DataWrapper(dataset),
                                batch_size=self.batch_size,
                                num_workers=self.n_workers,
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
        This method is used to generate class prediction given RGBD image

        :param img: img to generate class prediction
        :type img: engine.data.Image

        :return: predicted label
        :rtype: engine.target.Category

        """

        self._validate_x(img)

        self.model.to(torch.device(self.device))
        self.model.eval()

        img = img.convert("channels_first")
        img = np.expand_dims(img, 0)

        with torch.no_grad():
            tensor_img = torch.tensor(img, device=torch.device(self.device)).float()
            prob_prediction = torch.nn.functional.softmax(self.model(tensor_img).flatten(), dim=0)
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

        if isinstance(self.architecture, str):
            architecture = self.architecture
            is_builtin_architecture = True
        else:
            architecture = type(self.model).__name__
            is_builtin_architecture = False

        model_weight_file = os.path.join(path, 'model_weights.pt')
        metadata_file = os.path.join(path, 'metadata.json')

        metadata = {'framework': 'pytorch',
                    'model_paths': ['model_weights.pt'],
                    'format': 'pt',
                    'architecture': architecture,
                    'is_builtin_architecture': is_builtin_architecture,
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

        return True

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
        assert os.path.exists(metadata_file),\
            'Metadata file ("metadata.json") does not exist under the given path "{}"'.format(path)

        fid = open(metadata_file, 'r')
        metadata = json.load(fid)
        fid.close()

        if metadata['is_builtin_architecture'] and metadata['architecture'] != self.architecture:
            msg = 'The architecture specification of saved model does not match current model instance\n'
            msg += '`architecture` parameter of the saved model is: "{}"'.format(metadata['architecture'])
            msg += '`architecture` parameter of current model instance is: "{}"'.format(self.architecture)
            raise ValueError(msg)

        model_weight_file = os.path.join(path, metadata['model_paths'][0])
        assert os.path.exists(model_weight_file),\
            'Model weights "{}" does not exist'.format(model_weight_file)

        self.model.cpu()
        self.model.load_state_dict(torch.load(model_weight_file, map_location=torch.device('cpu')))

        if verbose:
            print('Pretrained model is loaded successfully')

    def download(self, path):
        """
        This function is used to download a pretrained model for the hand gesture recognition task
        Calling load(path) after this function will load the downloaded model weights

        :param path: path to the saved location. Under this path `model_weights.pt` and `metadata.json`
                     will be downloaded so different paths for different models should be given to avoid
                     overwriting previously downloaded model
        :type path: str
        """

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if isinstance(self.architecture, str) and self.architecture in PRETRAINED_MODEL:
            assert self.n_class == 16,\
                'For pretrained hand gesture model, `n_class` must be 16'

            server_url = os.path.join(OPENDR_SERVER_URL,
                                      'perception',
                                      'multimodal_human_centric',
                                      'rgbd_hand_gesture_learner')

            model_name = '{}_{}'.format('hand_gesture', self.architecture)

            metadata_url = os.path.join(server_url, '{}.json'.format(model_name))
            metadata_file = os.path.join(path, 'metadata.json')
            urlretrieve(metadata_url, metadata_file)

            weights_url = os.path.join(server_url, '{}.pt'.format(model_name))
            weights_file = os.path.join(path, 'model_weights.pt')
            urlretrieve(weights_url, weights_file)
            print('Pretrained model downloaded to the following directory\n{}'.format(path))
        else:
            raise UserWarning('No pretrained model for architecture "{}"'.format(self.architecture))

    def optimize(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
