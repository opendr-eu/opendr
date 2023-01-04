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

import argparse

from opendr.engine.datasets import ExternalDataset
from opendr.perception.object_tracking_2d import SiamRPNLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-root", help="Dataset root folder", type=str, default=".")

    args = parser.parse_args()

    learner = SiamRPNLearner(device=args.device)
    learner.download(".", mode="pretrained")
    learner.load("siamrpn_opendr")

    # download otb2015 dataset and run
    # alternatively you can download the "test_data" and use "OTBtest" dataset_type to only run on one small video
    learner.download(args.data_root, "otb2015", verbose=True, overwrite=False)
    dataset = ExternalDataset(args.data_root, dataset_type="OTB2015")
    learner.eval(dataset)
