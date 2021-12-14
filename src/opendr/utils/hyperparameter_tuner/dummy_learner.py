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

import torch
import torch.nn as nn
from opendr.engine.learners import Learner


class DummyLearner(Learner):
    """
    Dummy implementation of the Learner class. It is created for testing the hyperparameter tuner.
    """
    def __init__(self, lr=0.001, epochs=1, optimizer='SGD', out_features=12):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(DummyLearner, self).__init__(lr=lr, optimizer=optimizer)
        self.model = self._get_model(out_features)
        self.torch_optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr)
        self.n_epochs = epochs

    def fit(self, dataset, val_dataset=None, logging_path='', silent=False, verbose=False):
        loss_function = nn.MSELoss()
        x, y = self._get_data()
        for epoch in range(self.n_epochs):
            prediction = self.model(x)
            loss = loss_function(prediction.view(-1), y)
            self.torch_optimizer.zero_grad()
            loss.backward()
            self.torch_optimizer.step()

    def eval(self, dataset):
        loss_function = nn.MSELoss()
        self.model.eval()
        with torch.no_grad():
            x, y = self._get_data()
            prediction = self.model(x)
            loss = loss_function(prediction.view(-1), y).item()
        return loss

    @staticmethod
    def get_hyperparameters():
        hyperparameters = [
            {'name': 'optimizer', 'type': 'categorical', 'choices': ["Adam", "RMSprop", "SGD"]},
            {'name': 'lr', 'type': 'float', 'low': 0.00001, 'high': 0.01, 'log': True},
            {'name': 'out_features', 'type': 'int', 'low': 1, 'high': 12},
        ]
        return hyperparameters

    @staticmethod
    def get_objective_function():
        def objective_function(eval_stats):
            return eval_stats
        return objective_function

    @staticmethod
    def _get_model(out_features):
        layers = []
        in_features = 5
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(out_features, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _get_data():
        x = torch.rand((512, 5), dtype=torch.float32)
        y = torch.sin(x[:, 0]) * x[:, 1] ** 2 + torch.cos(x[:, 2]) * x[:, 3] ** 3 + x[:, 4]
        return x, y

    # All methods below are dummy implementations of the abstract methods that are inherited.
    def save(self, path):
        pass

    def infer(self):
        pass

    def load(self, path):
        pass

    def optimize(self, params):
        pass

    def reset(self):
        pass
