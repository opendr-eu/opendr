# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import shutil
import unittest
import warnings
import zipfile
from pathlib import Path
import numpy as np

from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner
from opendr.perception.continual_slam.datasets.kitti import KittiDataset
from opendr.perception.continual_slam.algorithm.depth_pose_module.replay_buffer import ReplayBuffer
from opendr.perception.continual_slam.algorithm.loop_closure.pose_graph_optimization import PoseGraphOptimization
from opendr.engine.data import Image


def rmfile(path):
    try:
        os.remove(path)
    except OSError as e:
        print(f'Error: {e.filename} - {e.strerror}.')


def rmdir(_dir):
    try:
        shutil.rmtree(_dir)
    except OSError as e:
        print(f'Error: {e.filename} - {e.strerror}.')


class TestContinualSlamLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************\nTEST Continaul SLAM Predictor/Learner\n******************************")
        cls.temp_dir = os.path.join('tests', 'sources', 'tools', 'perception', 'continual_slam',
                                    'continual_slam_temp')
        if os.path.exists(cls.temp_dir):
            rmdir(cls.temp_dir)
        os.makedirs(cls.temp_dir)

        # Download all required files for testing
        cls.model_weights = ContinualSLAMLearner.download(path=cls.temp_dir, trained_on="semantickitti")
        test_data_zipped = ContinualSLAMLearner.download(path=cls.temp_dir, mode="test_data")
        cls.test_data = os.path.join(cls.temp_dir, "test_data")
        with zipfile.ZipFile(test_data_zipped, "r") as f:
            f.extractall(cls.temp_dir)

        # Configuration for the weights pre-trained on SemanticKITTI
        cls.config_file = str(Path(sys.modules[
            ContinualSLAMLearner.__module__].__file__).parent / 'configs' / 'singlegpu_kitti.yaml')
        cls.config_file.depth_pose.load_weights_folder = cls.model_weights
        cls.config_file.dataset.dataset_path = cls.test_data

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(cls.temp_dir)

    def test_init(self):
        predictor = ContinualSLAMLearner(self.config_file, mode="predictor")
        learner = ContinualSLAMLearner(self.config_file, mode="learner")
        assert predictor.step == 0
        assert learner.step == 0

    def test_fit(self):
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)

        dataset = KittiDataset(self.config_file)
        learner = ContinualSLAMLearner(self.config_file, mode="learner")
        predictor = ContinualSLAMLearner(self.config_file, mode="predictor")
        replay_buffer = ReplayBuffer(buffer_size=5,
                                     save_memory=True,
                                     dataset_config_path=self.config_file,
                                     sample_size=3)
        # Test without replay buffer
        for item in dataset:
            replay_buffer.add(item)
            item = ContinualSLAMLearner._input_formatter(item)
            sample = [item]
            assert learner.fit(sample, learner=True)

        # Test with replay buffer
        for item in dataset:
            replay_buffer.add(item)
            if replay_buffer.size < 3:
                continue
            item = ContinualSLAMLearner._input_formatter(item)
            sample = replay_buffer.sample()
            sample.insert(0, item)
            assert learner.fit(sample, learner=True)
            try:
                predictor.fit(sample, learner=True)
                assert False, "Should raise NotImplementedError"
            except NotImplementedError:
                pass

    def test_infer(self):
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)

        dataset = KittiDataset(self.config_file)
        predictor = ContinualSLAMLearner(self.config_file, mode="predictor")
        learner = ContinualSLAMLearner(self.config_file, mode="learner")

        # Test without loop closure
        for item in dataset:
            (depth, odometry), losses = predictor.infer(item, return_losses=True)
            assert losses is None, "Losses should be None"
            self.assertIsInstance(depth, Image)
            self.assertIsInstance(odometry, np.array)
            try:
                learner.infer(item, return_losses=True)
                assert False, "Should raise NotImplementedError"
            except NotImplementedError:
                pass

        # Test with loop closure
        predictor.config_file.depth_pose.loop_closure = True
        for item in dataset:
            depth, odometry, losses, lc, pose_graph = predictor.infer(item, return_losses=True)
            assert losses is not None, "Losses should not be None"
            assert lc is False, "Loop closure should be False since we are only using first 50 frames"
            self.assertIsInstance(depth, Image)
            self.assertIsInstance(odometry, np.array)
            self.assertIsInstance(pose_graph, PoseGraphOptimization)

    def test_save(self):
        predictor = ContinualSLAMLearner(self.config_file, mode="predictor")
        temp_model_path = self.tempdir + '/test_save_weights'
        location = predictor.save(temp_model_path)
        self.assertTrue(temp_model_path == location)
        self.assertTrue(os.path.exists(os.path.join(temp_model_path, 'depth_decoder.pth')))
        self.assertTrue(os.path.exists(os.path.join(temp_model_path, 'pose_encoder.pth')))
        self.assertTrue(os.path.exists(os.path.join(temp_model_path, 'depth_encoder.pth')))
        self.assertTrue(os.path.exists(os.path.join(temp_model_path, 'pose_decoder.pth')))

    def test_load(self):
        predictor = ContinualSLAMLearner(self.config_file, mode="predictor")
        successful = predictor.load(self.model_weights)
        self.assertTrue(successful)

if __name__ == "__main__":
    unittest.main()
