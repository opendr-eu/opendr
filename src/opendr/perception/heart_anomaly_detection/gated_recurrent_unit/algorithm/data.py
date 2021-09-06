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

import pickle
import numpy as np
import torch
from urllib.request import urlretrieve
import os

# OepnDR imports
from opendr.engine.datasets import DatasetIterator
from opendr.engine.data import Timeseries
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL


class DataWrapper:
    def __init__(self, opendr_dataset):
        self.dataset = opendr_dataset

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset.__getitem__(i)
        x = torch.from_numpy(x.data).float()
        y = torch.tensor([y.data, ]).long()
        return x, y


class Dataset(DatasetIterator):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        x = self.x[i]
        y = int(self.y[i][0])
        return Timeseries(x), Category(y)


def get_AF_dataset(data_file, fold_idx, sample_length=30, standardize=True):
    assert sample_length >= 30, 'The `sample_length` of AF dataset must be at least 30 seconds'
    assert fold_idx in [0, 1, 2, 3, 4], '`fold_idx` must be from the set {0, 1, 2, 3, 4}'

    # download AF dataset
    server_url = os.path.join(OPENDR_SERVER_URL,
                              'perception',
                              'heart_anomaly_detection',
                              'data',
                              'AF.pickle')

    urlretrieve(server_url, data_file)

    # read data file
    fid = open(data_file, 'rb')
    data = pickle.load(fid)
    fid.close()

    # prepare data fold
    train_indices = data['train_indices'][fold_idx]
    test_indices = data['test_indices'][fold_idx]
    x = data['x']
    y = data['y']
    nb_train = len(train_indices)
    nb_test = len(test_indices)
    length = data['sampling_frequency'] * sample_length

    x_train = np.zeros((nb_train, 1, length), dtype=np.float32)
    y_train = np.asarray(y)[train_indices] - 1

    x_test = np.zeros((nb_test, 1, length), dtype=np.float32)
    y_test = np.asarray(y)[test_indices] - 1

    for count, idx in enumerate(train_indices):
        x_train[count, 0, :] = pad_sample(x[idx], length)

    for count, idx in enumerate(test_indices):
        x_test[count, 0, :] = pad_sample(x[idx], length)

    c0 = np.where(y_train == 0)[0].size
    c1 = np.where(y_train == 1)[0].size
    c2 = np.where(y_train == 2)[0].size
    c3 = np.where(y_train == 3)[0].size

    class_weight = [1e3/float(c0), 1e3/float(c1), 1e3/float(c2), 1e3/float(c3)]

    if standardize:
        x_tmp = x_train.flatten()
        x_tmp = x_tmp[np.where(x_tmp != 0)]
        mean = np.reshape(np.mean(x_tmp), (1, 1, 1))
        std = np.reshape(np.std(x_tmp), (1, 1, 1))
        if std[0, 0, 0] < 1e-5:
            std[0, 0, 0] = 1.0

        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    train_set = Dataset(x_train, y_train)
    test_set = Dataset(x_test, y_test)

    return train_set, test_set, length, class_weight


def pad_sample(x, size):
    x = x.flatten()
    if x.size >= size:
        y = x[:size]
    else:
        y = np.zeros((size,), dtype=np.float32)
        y[:x.size] = x

    return y
