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

import numpy as np


# Taken from P2B: https://github.com/HaozheQi/P2B
class Success(object):
    """Computes and stores the Success"""

    def __init__(self, n=21, max_overlap=1):
        self.max_overlap = max_overlap
        self.Xaxis = np.linspace(0, self.max_overlap, n)
        self.reset()

    def reset(self):
        self.overlaps = []

    def add_overlap(self, val):
        self.overlaps.append(val)

    @property
    def count(self):
        return len(self.overlaps)

    @property
    def value(self):
        succ = [
            np.sum(i >= thres for i in self.overlaps).astype(float) /
            self.count
            for thres in self.Xaxis
        ]
        return np.array(succ)

    @property
    def average(self):
        if len(self.overlaps) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_overlap


# Taken from P2B: https://github.com/HaozheQi/P2B
class Precision(object):
    """Computes and stores the Precision"""

    def __init__(self, n=21, max_accuracy=2):
        self.max_accuracy = max_accuracy
        self.Xaxis = np.linspace(0, self.max_accuracy, n)
        self.reset()

    def reset(self):
        self.accuracies = []

    def add_accuracy(self, val):
        self.accuracies.append(val)

    @property
    def count(self):
        return len(self.accuracies)

    @property
    def value(self):
        prec = [
            np.sum(i <= thres for i in self.accuracies).astype(float) /
            self.count
            for thres in self.Xaxis
        ]
        return np.array(prec)

    @property
    def average(self):
        if len(self.accuracies) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_accuracy
