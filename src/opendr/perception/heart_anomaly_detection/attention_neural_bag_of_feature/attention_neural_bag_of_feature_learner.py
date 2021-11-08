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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Timeseries
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR imports
from opendr.perception.heart_anomaly_detection.gated_recurrent_unit.algorithm import (
    DataWrapper,
    get_AF_dataset
)
from opendr.perception.heart_anomaly_detection.gated_recurrent_unit.gated_recurrent_unit_learner import (
    ClassifierTrainer,
    get_cosine_lr_scheduler,
    get_multiplicative_lr_scheduler
)
from opendr.perception.heart_anomaly_detection.attention_neural_bag_of_feature.algorithm import models

__all__ = ['get_AF_dataset',
           'get_cosine_lr_scheduler',
           'get_multiplicative_lr_scheduler']

PRETRAINED_SAMPLE_LENGTH = [30]
PRETRAINED_N_CODEWORD = [256, 512]
PRETRAINED_QUANT_TYPE = ['nbof', ]
PRETRAINED_ATTENTION_TYPE = ['temporal', ]
AF_SAMPLING_RATE = 300


class AttentionNeuralBagOfFeatureLearner(Learner):
    def __init__(self,
                 in_channels,
                 series_length,
                 n_class,
                 n_codeword=256,
                 quantization_type='nbof',
                 attention_type='spatial',
                 lr_scheduler=get_cosine_lr_scheduler(1e-3, 1e-5),
                 optimizer='adam',
                 weight_decay=0.0,
                 dropout=0.2,
                 iters=300,
                 batch_size=32,
                 checkpoint_after_iter=1,
                 checkpoint_load_iter=0,
                 temp_path='',
                 device='cpu',
                 test_mode=False,
                 ):

        super(AttentionNeuralBagOfFeatureLearner, self).__init__(batch_size=batch_size,
                                                                 checkpoint_after_iter=checkpoint_after_iter,
                                                                 checkpoint_load_iter=checkpoint_load_iter,
                                                                 temp_path=temp_path,
                                                                 device=device)

        assert quantization_type in ['nbof', 'tnbof'],\
            'Parameter `quantization_type` must be "nbof" or "tnbof"\n' +\
            'Provided value: {}'.format(quantization_type)

        assert attention_type in ['spatial', 'temporal'],\
            'Parameter `attention_type` must be "spatial" or "temporal"\n' +\
            'Provided value: {}'.format(attention_type)

        assert checkpoint_load_iter < iters,\
            '`check_point_load_iter` must be less than `iters`\n' +\
            'Given check_point_load_iter={} and iters={}'.format(checkpoint_load_iter, iters)

        assert optimizer in ['adam', 'sgd'],\
            'given optimizer "{}" is not supported, please select set optimizer to "adam" or "sgd"'.format(optimizer)

        self.in_channels = in_channels
        self.series_length = series_length
        self.n_codeword = n_codeword
        self.quantization_type = quantization_type
        self.attention_type = attention_type
        self.n_class = n_class
        self.lr_scheduler = lr_scheduler
        self.n_epoch = iters
        self.batch_size = batch_size
        self.checkpoint_freq = checkpoint_after_iter
        self.epoch_idx = checkpoint_load_iter
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.test_mode = test_mode
        self.temp_path = temp_path

        if quantization_type == 'nbof':
            self.model = models.ANBoF(in_channels, series_length, n_codeword, attention_type, n_class, dropout)
        else:
            self.model = models.ATNBoF(in_channels, series_length, n_codeword, attention_type, n_class, dropout)

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

    def _validate_dataset(self, dataset):
        """
        This internal function is used to perform basic validation of the data dimensions
        """

        if dataset is None:
            return

        x, y = dataset.__getitem__(0)

        assert isinstance(x, Timeseries),\
            'The first element returned by __getitem__ must be an instance of engine.data.Timeseries class\n' +\
            'Received an instance of type: {}'.format(type(x))

        assert isinstance(y, Category),\
            'The second element returned by __getitem__ must be an instance of engine.target.Category class\n' +\
            'Received an instance of type: {}'.format(type(y))

        x = x.numpy()
        assert x.shape[0] == self.in_channels,\
            'Parameter `in_channels` provided during initialization does not match ' +\
            'the first dimension of series generated by dataset\n' +\
            'Parameter `in_channels` provided during model initialization: {}\n'.format(self.in_channels) +\
            'First dimension of the series generated by dataset: {}\n'.format(x.shape[0])

        assert x.shape[1] == self.series_length,\
            'Parameter `series_length` provided during initialization does not match ' +\
            'the second dimension of series generated by dataset\n' +\
            'Parameter `series_length` provided during model initialization: {}\n'.format(self.series_length) +\
            'Second dimension of the series generated by dataset: {}\n'.format(x.shape[1])

    def fit(self,
            train_set,
            val_set=None,
            test_set=None,
            class_weight=None,
            logging_path='',
            silent=False,
            verbose=True):
        """
        Method to train the attention neural bag-of-features model

        :param train_set: object that holds training samples
        :type train_set: engine.datasets.DatasetIterator object
        :param val_set: object that holds the validation samples, default to None
                        if available, `val_set` is used to select the best checkpoint for final model
        :type val_set: engine.datasets.DatasetIterator object
        :param test_set: object that holds the test samples, default to None
        :type test_set: engine.datasets.DatasetIterator object
        :param class_weight: the weight used to alleviate imbalanced classes, should be a list having
                        the length equal the number of classes
        :type class_weight: list
        :param logging_path: path to save tensoboard data, default to ""
        :type logging_path: string
        :param silent: disable performance printing, default to False
        :type silent: bool
        :param verbose: enable the progress bar of each epoch, default to True
        :type verbose: bool

        :return: returns the performance curves
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

        trainer = ClassifierTrainer(n_epoch=self.n_epoch,
                                    epoch_idx=self.epoch_idx,
                                    lr_scheduler=self.lr_scheduler,
                                    class_weight=class_weight,
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
        :type dataset: engine.datasets.DatasetIterator object
        :return: a dictionary that contains `cross_entropy`, `acc`, `precision`, `recall`, `f1` as keys
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

        if verbose:
            loader = tqdm(loader, desc='#Evaluating ', ncols=80, ascii=True)

        L = torch.nn.CrossEntropyLoss()
        n_sample = 0
        loss = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long().flatten()

                predictions = self.model(inputs)
                n_sample += inputs.size(0)
                loss += L(predictions, targets).item()
                y_true.extend(targets.flatten().tolist())
                y_pred.extend(predictions.argmax(dim=-1).flatten().tolist())

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        metrics = {'cross_entropy': loss / n_sample,
                   'acc': acc,
                   'precision': precision,
                   'recall': recall,
                   'f1': f1}

        if not silent:
            print('cross_entropy: {:.6f}'.format(metrics['cross_entropy']))
            print('acc: {:.6f}'.format(metrics['acc']))
            print('precision: {:.6f}'.format(metrics['precision']))
            print('recall: {:.6f}'.format(metrics['recall']))
            print('f1: {:.6f}'.format(metrics['f1']))

        return metrics

    def infer(self, series):
        """
        This method is used to generate class prediction given a time-series

        :param series: the input series to generate class prediction
        :type series: engine.data.Timeseries

        """

        assert isinstance(series, Timeseries),\
            'Input to `infer()` must be an instance of engine.data.Timeseries\n' +\
            'Received an instance of type: {}'.format(type(series))

        self.model.to(torch.device(self.device))
        self.model.eval()

        series = series.numpy()

        assert series.shape[0] == self.in_channels,\
            'Parameter `in_channels` provided during initialization does not match ' +\
            'the first dimension of the input series\n' +\
            'Parameter `in_channels` provided during model initialization: {}\n'.format(self.in_channels) +\
            'First dimension of the input series : {}\n'.format(series.shape[0])

        assert series.shape[1] == self.series_length,\
            'Parameter `series_length` provided during initialization does not match ' +\
            'the second dimension of input series\n' +\
            'Parameter `series_length` provided during model initialization: {}\n'.format(self.series_length) +\
            'Second dimension of the input series: {}\n'.format(series.shape[1])

        series = np.expand_dims(series, 0)

        series = torch.tensor(series, device=torch.device(self.device)).float()
        prob_prediction = torch.nn.functional.softmax(self.model(series).flatten(), dim=0)
        class_prediction = prob_prediction.argmax().cpu().item()
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

        model_weight_file = os.path.join(path, 'model_weights.pt')
        metadata_file = os.path.join(path, 'metadata.json')

        metadata = {'framework': 'pytorch',
                    'format': 'pt',
                    'quantization_type': self.quantization_type,
                    'attention_type': self.attention_type,
                    'in_channels': self.in_channels,
                    'series_length': self.series_length,
                    'n_codeword': self.n_codeword,
                    }

        try:
            torch.save(self.model.state_dict(), model_weight_file)
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
        model_weight_file = os.path.join(path, 'model_weights.pt')

        assert os.path.exists(metadata_file),\
            'Metadata file ("metadata.json") does not exist under the given path "{}"'.format(path)

        assert os.path.exists(model_weight_file),\
            'Model weights ("model_weights.pt") does not exist under the given path "{}"'.format(path)

        fid = open(metadata_file, 'r')
        metadata = json.load(fid)
        fid.close()

        assert metadata['in_channels'] == self.in_channels,\
            'Parameter `in_channels` provided during initialization does not match ' +\
            'parameter `in_channels` of the saved model\n' +\
            'Parameter `in_channels` provided during model initialization: {}\n'.format(self.in_channels) +\
            'Parameter `in_channels` of the saved model: {}\n'.format(metadata['in_channels'])

        assert metadata['series_length'] == self.series_length,\
            'Parameter `series_length` provided during initialization does not match ' +\
            'parameter `series_length` of the saved model\n' +\
            'Parameter `series_length` provided during model initialization: {}\n'.format(self.series_length) +\
            'Parameter `series_length` of the saved model: {}\n'.format(metadata['series_length'])

        assert metadata['n_codeword'] == self.n_codeword,\
            'Parameter `n_codeword` provided during initialization does not match ' +\
            'parameter `n_codeword` of the saved model\n' +\
            'Parameter `n_codeword` provided during model initialization: {}\n'.format(self.n_codeword) +\
            'Parameter `n_codeword` of the saved model: {}\n'.format(metadata['n_codeword'])

        assert metadata['quantization_type'] == self.quantization_type,\
            'Parameter `quantization_type` provided during initialization does not match ' +\
            'parameter `quantization_type` of the saved model\n' +\
            'Parameter `quantization_type` provided during model initialization: {}\n'.format(self.quantization_type) +\
            'Parameter `quantization_type` of the saved model: {}\n'.format(metadata['quantization_type'])

        assert metadata['attention_type'] == self.attention_type,\
            'Parameter `attention_type` provided during initialization does not match ' +\
            'parameter `attention_type` of the saved model\n' +\
            'Parameter `attention_type` provided during model initialization: {}\n'.format(self.attention_type) +\
            'Parameter `attention_type` of the saved model: {}\n'.format(metadata['attention_type'])

        self.model.cpu()
        self.model.load_state_dict(torch.load(model_weight_file, map_location=torch.device('cpu')))

        if verbose:
            print('Pretrained model is loaded successfully')

    def download(self, path, fold_idx):
        """
        This function is used to download a pretrained model given the current model specification
        Calling load(path) after this function will load the downloaded model weights

        :param path: path to the saved location. Under this path `model_weights.pt` and `metadata.json`
                     will be downloaded so different paths for different models should be given to avoid
                     overwriting previously downloaded model
        :type path: str
        :param fold_idx: index of the cross-validation fold. Pretrained models for 5-fold cross-validation
                         are available
        :type fold_idx: int
        """

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        assert self.n_class == 4,\
            'Only support pretrained model for the AF dataset, which has 4 classes.\n' +\
            'Current model specification has {} classes'.format(self.n_class)

        assert fold_idx in [0, 1, 2, 3, 4],\
            '`fold_idx` must receive a value from the list [0, 1, 2, 3, 4]\n' +\
            'provided value: {}'.format(fold_idx)

        sample_length = int(self.series_length / AF_SAMPLING_RATE)
        assert sample_length in PRETRAINED_SAMPLE_LENGTH,\
            'Current `series_length` does not match supported `series_length`.' +\
            'Supported values of `series_length` includes\n' +\
            '\n'.join([str(v * AF_SAMPLING_RATE) for v in PRETRAINED_SAMPLE_LENGTH])

        assert self.in_channels == 1,\
            'The value of `in_channels` parameter must be 1.\n' +\
            'Provided value of `in_channels`: {}'.format(self.in_channels)

        assert self.n_codeword in PRETRAINED_N_CODEWORD,\
            'Current `n_codeword` does not match supported `n_codeword`.' +\
            'Supported values of `n_codeword` includes\n' +\
            '\n'.join([str(v) for v in PRETRAINED_N_CODEWORD])

        assert self.quantization_type in PRETRAINED_QUANT_TYPE,\
            'Current `quantization_type` does not match supported `quantization_type`.' +\
            'Supported values of `quantization_type` includes\n' +\
            '\n'.join([str(v) for v in PRETRAINED_QUANT_TYPE])

        assert self.attention_type in PRETRAINED_ATTENTION_TYPE,\
            'Current `attention_type` does not match supported `attention_type`.' +\
            'Supported values of `attention_type` includes\n' +\
            '\n'.join([str(v) for v in PRETRAINED_ATTENTION_TYPE])

        server_url = os.path.join(OPENDR_SERVER_URL,
                                  'perception',
                                  'heart_anomaly_detection',
                                  'attention_neural_bag_of_feature')

        model_name = 'AF_{}_{}_{}_{}_{}'.format(self.quantization_type,
                                                self.attention_type,
                                                fold_idx,
                                                sample_length,
                                                self.n_codeword)

        metadata_url = os.path.join(server_url, '{}.json'.format(model_name))
        metadata_file = os.path.join(path, 'metadata.json')
        urlretrieve(metadata_url, metadata_file)

        weights_url = os.path.join(server_url, '{}.pt'.format(model_name))
        weights_file = os.path.join(path, 'model_weights.pt')
        urlretrieve(weights_url, weights_file)
        print('Pretrained model downloaded to the following directory\n{}'.format(path))

    def optimize(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
