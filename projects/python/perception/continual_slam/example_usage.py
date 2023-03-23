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
import os
import cv2

from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner
from opendr.perception.continual_slam.datasets.kitti import KittiDataset
from opendr.perception.continual_slam.algorithm.depth_pose_module.replay_buffer import ReplayBuffer


def download_models(root_path):
    if os.path.exists(root_path):
        return os.path.join(root_path, 'models')
    model_path = ContinualSLAMLearner.download(path=root_path, trained_on="cityscapes")
    return model_path


def download_test_data(root_path):
    if os.path.exists(root_path):
        return os.path.join(root_path, 'test_data', 'infer_data')
    dataset_path = ContinualSLAMLearner.download(path=root_path, mode="test_data")
    return dataset_path


def load_model(model_path, model: ContinualSLAMLearner):
    model.load(model_path)
    return model


def train(dataset_path, config_file, model_path):
    dataset = KittiDataset(str(dataset_path), config_file)
    learner = ContinualSLAMLearner(config_file, mode="learner")
    learner = load_model(model_path, learner)
    replay_buffer = ReplayBuffer(buffer_size=5,
                                 save_memory=True,
                                 dataset_config_path=config_file,
                                 sample_size=3)
    for item in dataset:
        replay_buffer.add(item)
        if replay_buffer.count < 3:
            continue
        item = ContinualSLAMLearner._input_formatter(item)
        sample = replay_buffer.sample()
        sample.insert(0, item)
        learner.fit(sample, learner=True)
    return (learner.save("./cl_slam/trained_model/"))


def inference(dataset_path, config_file, model_path=None):
    dataset = KittiDataset(str(dataset_path), config_file)
    predictor = ContinualSLAMLearner(config_file, mode="predictor")
    load_model(model_path, predictor)
    predictor.do_loop_closure = True
    for i, item in enumerate(dataset):
        depth, odometry, losses, lc, pose_graph = predictor.infer(item, return_losses=True)
        cv2.imwrite(f'./cl_slam/predictions/depth_{i}.png', depth.opencv())

def main():
    env = os.getenv('OPENDR_HOME')
    config_file = os.path.join(env, 'src/opendr/perception/continual_slam/configs/singlegpu_kitti.yaml')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_root', type=str, default='./cl_slam/data',
                        help='To specify the path to the data root directory, where models are stored')
    parser.add_argument('-m', '--model_root', type=str, default='./cl_slam/model',
                        help='To specify the path where models are going to be downloaded stored')

    args = parser.parse_args()

    model_path = download_models(args.model_root)
    data_path = download_test_data(args.data_root)
    trained_model = train(data_path, config_file, model_path)
    print('-' * 40 + '\n===> Training succeeded\n' + '-' * 40)
    inference(data_path, config_file, trained_model)
    print('-' * 40 + '\n===> Inference succeeded\n' + '-' * 40)


if __name__ == "__main__":
    main()
