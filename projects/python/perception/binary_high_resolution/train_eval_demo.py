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

from opendr.perception.binary_high_resolution import BinaryHighResolutionLearner, visualize
from opendr.engine.datasets import ExternalDataset

import torch
import numpy as np
torch.random.manual_seed(1)
np.random.seed(1)


if __name__ == '__main__':
    # Prepare the dataset loader
    dataset = ExternalDataset("./demo_dataset", "VOC2012")

    learner = BinaryHighResolutionLearner(device='cuda')
    # Fit the learner
    learner.fit(dataset)
    # Save the trained model
    learner.save("test_model")
    # Visualize the results
    visualize(learner, "./demo_dataset/test_img.png")
    print("Evaluation results = ", learner.eval(dataset))
