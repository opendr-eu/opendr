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

import json
import logging
import os
from urllib.request import urlretrieve
from urllib.error import URLError

import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.data import Timeseries
from opendr.engine.learners import Learner
from opendr.engine.target import Category
from opendr.perception.speech_recognition.quadraticselfonn.algorithm.audioutils import get_mfcc
from opendr.perception.speech_recognition.quadraticselfonn.algorithm.model import QuadraticSelfOnnNet


class QuadraticSelfOnnLearner(Learner):
    def __init__(self,
                 lr=0.01,
                 iters=30,
                 batch_size=64,
                 optimizer='sgd',
                 checkpoint_after_iter=0,
                 checkpoint_load_iter=0,
                 temp_path='temp',
                 device='cuda',
                 expansion_order=3,
                 output_classes_n=20,
                 momentum=0.9,
                 preprocess_to_mfcc=True,
                 sample_rate=16000
                 ):
        super(QuadraticSelfOnnLearner, self).__init__(lr=lr, iters=iters, batch_size=batch_size,
                                                      optimizer=optimizer,
                                                      checkpoint_after_iter=checkpoint_after_iter,
                                                      checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                                      device=device)
        self.logger = logging.getLogger("QuadraticSelfOnnLearner")
        self.momentum = momentum
        self.preprocess_to_mfcc = preprocess_to_mfcc
        self.sample_rate = sample_rate
        self.output_classes_n = output_classes_n
        self.expansion_order = expansion_order

        self.model = QuadraticSelfOnnNet(self.output_classes_n, q=self.expansion_order)
        self.loss = nn.NLLLoss()

        self.model.to(self.device)
        self.loss.to(self.device)

        if self.optimizer == "sgd":
            self.optimizer_func = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == "adam":
            self.optimizer_func = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.logger.warning("Only SGD and Adam optimizers are available for this method.")
            self.optimizer_func = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    @property
    def expansion_order(self):
        return self._expansion_order

    @expansion_order.setter
    def expansion_order(self, value):
        if type(value) is not int or value < 1:
            raise TypeError("The expansion order should be a positive integer")
        else:
            self._expansion_order = value

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, value):
        if type(value) is not float or value < 0:
            raise TypeError("Momentum should be a float and non-negative")
        else:
            self._momentum = value

    @property
    def output_classes_n(self):
        return self._output_classes_n

    @output_classes_n.setter
    def output_classes_n(self, value):
        if type(value) is not int or value < 2:
            raise TypeError("The amount of target classes should be an int and greater than or equal to 2")
        else:
            self._output_classes_n = value

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if type(value) is not int or value <= 0:
            raise TypeError("Sample rate should be an integer and positive")
        else:
            self._sample_rate = value

    @property
    def preprocess_to_mfcc(self):
        return self._preprocess_to_mfcc

    @preprocess_to_mfcc.setter
    def preprocess_to_mfcc(self, value):
        if type(value) is not bool:
            raise TypeError("Preprocessing to MFCC should be a boolean")
        else:
            self._preprocess_to_mfcc = value

    def _signal_to_mfcc(self, signal):
        mfcc = np.apply_along_axis(
            lambda sample: get_mfcc(sample, self.sample_rate, n_mfcc=20, length=34),
            axis=1,
            arr=signal)
        return mfcc

    def _get_model_output(self, x):
        if self.preprocess_to_mfcc:
            x = self._signal_to_mfcc(x)
        x = t.Tensor(x)
        x = x.unsqueeze(1).to(self.device)
        predictions = self.model(x)
        return predictions

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=self.device == "cuda", shuffle=True)
        if not self.checkpoint_load_iter == 0:
            checkpoint_filename = os.path.join(
                self.temp_path + f"QuadraticSelfONN-{self.checkpoint_load_iter}.pth")
            if os.path.exists(checkpoint_filename):
                self.load(checkpoint_filename)
            else:
                print(f"No loadable checkpoint found for iteration {self.checkpoint_load_iter}")
        self.model.train()
        statistics = {}
        for epoch in range(1, self.iters + 1):
            if not silent:
                logging.info(f"Epoch {epoch}")
            statistics[epoch] = {"batch_losses": []}
            for batch_id, (x, y) in enumerate(dataloader):
                self.optimizer_func.zero_grad()
                output = self._get_model_output(x)
                y = y.to(self.device)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer_func.step()
                statistics[epoch]["batch_losses"].append(loss.data.item())
                if verbose and not silent:
                    logging.info(f"Batch {batch_id}: training loss {loss.data.item():.7}")
            if val_dataset is not None:
                statistics[epoch]["validation_results"] = self.eval(val_dataset)
                if not silent:
                    logging.info(f"Epoch {epoch} validation results:\n"
                                 f"Accuracy: {statistics[epoch]['validation_results']['test_accuracy']:.4}\n"
                                 f"Total loss: {statistics[epoch]['validation_results']['test_total_loss']:.7}")
            if not self.checkpoint_after_iter == 0 and epoch % self.checkpoint_after_iter == 0:
                filename = os.path.join(self.temp_path + f"QuadraticSelfONN-{epoch}.pth")
                self.save(filename)
                if not silent:
                    logging.info(f"Saved checkpoint to {filename}")

        return statistics

    def eval(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=self.device == "cuda")
        self.model.eval()
        test_loss = 0
        correct_predictions = 0
        for batch_id, (x, y) in enumerate(dataloader):
            output = self._get_model_output(x)
            y = y.to(self.device)
            test_loss += self.loss(output, y).data.item()
            predictions = output.max(1, keepdim=True)[1]
            correct_predictions += predictions.eq(y.view_as(predictions)).sum().item()
        return {"test_accuracy": correct_predictions / len(dataset),
                "test_total_loss": test_loss}

    def infer(self, batch):
        self.model.eval()
        if isinstance(batch, list):
            data = np.vstack([series.numpy() for series in batch])
        elif isinstance(batch, Timeseries):
            data = batch.numpy()
        else:
            raise TypeError("infer requires Timeseries or list of Timeseries as input.")
        output = self._get_model_output(data)
        prediction = output.max(1, keepdim=True)
        batch_predictions = []
        for target, confidence in zip(prediction[1], prediction[0].exp()):
            batch_predictions.append(Category(target.item(), confidence=confidence.item()))
        return batch_predictions[0] if len(batch_predictions) == 1 else batch_predictions

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        folder_basename = os.path.basename(path)
        model_path = os.path.join(path, folder_basename + ".pt")

        metadata = {"model_paths": [folder_basename + ".pt"],
                    "framework": "pytorch",
                    "format": "pt",
                    "has_data": False,
                    "inference_params": {"sample_rate": self.sample_rate},
                    "optimized": False,
                    "optimizer_info": {}}
        t.save(self.model.state_dict(), model_path)
        with open(os.path.join(path, folder_basename + ".json"), "w") as jsonfile:
            json.dump(metadata, jsonfile)
        return True

    def load(self, path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Could not find directory {path}")

        folder_basename = os.path.basename(path)
        with open(os.path.join(path, folder_basename + ".json")) as jsonfile:
            metadata = json.load(jsonfile)

        model_filename = os.path.basename(metadata["model_paths"][0])
        if "\\" in model_filename:
            model_filename = model_filename.split("\\")[-1]
        self.model.load_state_dict(t.load(os.path.join(path, model_filename)))
        self.model.eval()

    def optimize(self, **kwargs):
        pass

    def reset(self):
        for module in self.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def download_pretrained(self, path="."):
        target_directory = os.path.join(path, "QuadraticSelfOnn")
        jsonurl = OPENDR_SERVER_URL + "perception/speech_recognition/QuadraticSelfOnn/QuadraticSelfOnn.json"
        pturl = OPENDR_SERVER_URL + "perception/speech_recognition/QuadraticSelfOnn/QuadraticSelfOnn.pt"
        if not os.path.exists(target_directory):
            os.makedirs(target_directory, exist_ok=True)
        try:
            urlretrieve(jsonurl, os.path.join(target_directory, "QuadraticSelfOnn.json"))
            urlretrieve(pturl, os.path.join(target_directory, "QuadraticSelfOnn.pt"))
        except URLError as e:
            print("Could not retrieve pretrained model files!")
            raise e
