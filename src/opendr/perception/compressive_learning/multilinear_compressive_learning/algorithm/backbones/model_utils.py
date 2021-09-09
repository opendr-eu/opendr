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

import os
import pickle
from urllib.request import urlretrieve


def get_cifar_pretrained_weights(model_name):
    if model_name == '':
        return

    home_dir = os.path.expanduser('~')
    model_dir = os.path.join(home_dir,
                             '.cache',
                             'opendr',
                             'checkpoints',
                             'perception',
                             'compressive_learning',
                             'backbone')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_file = os.path.join(model_dir, '{}.pickle'.format(model_name))

    if not os.path.exists(model_file):
        server_url = 'ftp://opendrdata.csd.auth.gr/perception/compressive_learning/backbone/'
        model_url = os.path.join(server_url, '{}.pickle'.format(model_name))
        urlretrieve(model_url, model_file)
        print('Pretrained backbone model downloaded')

    fid = open(model_file, 'rb')
    state_dict = pickle.load(fid)['state_dict']
    fid.close()

    return state_dict
