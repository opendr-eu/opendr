
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
from engine.datasets import DatasetIterator
from engine.data import PointCloud


class PointCloudsDatasetIterator(DatasetIterator):
    def __init__(self, path, num_point_features=4):
        super().__init__()

        self.path = path
        self.num_point_features = num_point_features
        self.files = os.listdir(path)

    def __getitem__(self, idx):
        data = np.fromfile(
            str(self.path + "/" + self.files[idx]), dtype=np.float32, count=-1
        ).reshape([-1, self.num_point_features])

        return PointCloud(data)

    def __len__(self):
        return len(self.files)
