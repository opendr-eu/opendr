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

import sys
import os
from pathlib import Path

from opendr.engine.data import Image
from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner
from opendr.perception.continual_slam.datasets.kitti import KittiDataset
from opendr.perception.continual_slam.algorithm.depth_pose_module.replay_buffer import ReplayBuffer


def download_models():
    ContinualSLAMLearner.download(path="models", trained_on="semantickitti")

def download_test_data():
    dataset_path = ContinualSLAMLearner.download(path="test_data", mode="test_data")
    return dataset_path

def train(dataset_path, config_file):
    dataset = KittiDataset(dataset_path, config_file)
    learner = ContinualSLAMLearner(config_file, mode="learner")

    replay_buffer = ReplayBuffer(buffer_size=5,
                                save_memory=True,
                                dataset_config_path=config_file,
                                sample_size=3)
    for item in dataset:
        replay_buffer.add(item)
        if replay_buffer.size < 3:
            continue
        item = ContinualSLAMLearner._input_formatter(item)
        sample = replay_buffer.sample()
        sample.insert(0, item)
        learner.fit(sample, learner=True)

def inference(dataset_path, config_file):
    dataset = KittiDataset(dataset_path, config_file)
    predictor = ContinualSLAMLearner(config_file, mode="predictor")
    predictor.config_file.depth_pose.loop_closure = True
    for item in dataset:
        depth, odometry, losses, lc, pose_graph = predictor.infer(item, return_losses=True)
    # TODO: Ask Niclas if we need to output a visualization here or not

def main():
    env = os.getenv('OPENDR_HOME')
    config_file = os.path.join(env, 'src/opendr/perception/continual_slam/configs/singlegpu_kitti.yaml')

    download_models()
    dataset_path = download_test_data()
    train(dataset_path=dataset_path, config_file_path=config_file)
    inference()

if __name__ == "__main__":
    main()