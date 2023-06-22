#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020-2023 OpenDR European Project
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
import numpy as np
import torch
import argparse
from opendr.perception.heart_anomaly_detection import GatedRecurrentUnitLearner, \
 AttentionNeuralBagOfFeatureLearner, get_AF_dataset
from opendr.engine.data import Timeseries


if __name__ == '__main__':
    # Select the device for running
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('-model', type=str, help='model to be used for prediction: anbof or gru', required=True)
    parser.add_argument('--input_data', type=str, help="Path to input data or 'AF'", default='AF')
    parser.add_argument('--channels', type=int, help='Number of input channels. Default is 1.', default=1)
    parser.add_argument('--series_length', type=int, help='Length of input sequence. Default is 9000.', default=9000)
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint', default='./checkpoint')
    parser.add_argument('--n_class', type=int, help='Number of classes', default=4)
    parser.add_argument('--att_type', type=str, help='Attention type for ANBOF model.', default='temporal')

    args = parser.parse_args()

    # create a learner
    if args.model == 'gru':
        learner = GatedRecurrentUnitLearner(in_channels=args.channels, series_length=args.series_length,
                                            n_class=args.n_class, device=device)
    elif args.model == 'anbof':
        learner = AttentionNeuralBagOfFeatureLearner(in_channels=args.channels, series_length=args.series_length,
                                                     n_class=args.n_class, device=device, attention_type=args.att_type)
    # load the checkpoint
    if not os.path.exists(args.checkpoint):
        learner.download(path=args.checkpoint, fold_idx=0)
    else:
        learner.load(path=args.checkpoint)

    # load data and predict
    if args.input_data == 'AF':
        train_set, val_set, series_length, class_weight = get_AF_dataset(data_file='AF.dat', fold_idx=0, sample_length=30)
        learner.eval(val_set)
    else:
        data = Timeseries(np.load(args.input_data))
        prediction = learner.infer(data)
        print(prediction)
