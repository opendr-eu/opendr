# Copyright 2020-2022 OpenDR European Project
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
from pathlib import Path
import shutil
import unittest
import warnings
import zipfile

import numpy as np

from opendr.engine.data import PointCloud, Image
from opendr.engine.target import Heatmap
from opendr.perception.panoptic_segmentation import EfficientLpsLearner, SemanticKittiDataset


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


class TestEfficientLpsLearner(unittest.TestCase):
    temp_dir = None

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST EfficientLPS Learner\n**********************************")

        cls.temp_dir = Path(__file__).parent / "efficient_lps_temp"
        if cls.temp_dir.exists():
            rmdir(cls.temp_dir)
        cls.temp_dir.mkdir(parents=False, exist_ok=True)

        # Download all required files for testing
        cls.model_weights = EfficientLpsLearner.download(path=cls.temp_dir, trained_on="kitti")
        test_data_zipped = EfficientLpsLearner.download(path=cls.temp_dir, mode="test_data")

        cls.test_data = cls.temp_dir / "test_data"
        with zipfile.ZipFile(test_data_zipped, "r") as f:
            f.extractall(cls.test_data)

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(cls.temp_dir)

    def test_init(self):
        # Verify that the internal variables are initialized as expected by the other functions
        learner = EfficientLpsLearner()
        self.assertFalse(learner._is_model_trained)

    def test_fit(self):
        pass

    def test_eval(self):
        # From mmdet base code
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)

        val_dataset = SemanticKittiDataset(path=self.test_data / "eval_data", split="valid")
        learner = EfficientLpsLearner(batch_size=1)
        learner.load(self.model_weights)
        eval_results = learner.eval(val_dataset)
        self.assertIsInstance(eval_results, dict)

    def test_infer_single_point_cloud(self):
        # TODO: Verify Test file after uploading to server
        point_cloud_filename = self.test_data / "infer_data" / "seq08_f000000.bin"
        point_cloud = SemanticKittiDataset.load_point_cloud(point_cloud_filename)
        learner = EfficientLpsLearner()
        learner.load(self.model_weights)
        prediction = learner.infer(point_cloud, projected=True)
        self.assertIsInstance(prediction[0], Heatmap)
        self.assertIsInstance(prediction[1], Heatmap)
        self.assertIsInstance(prediction[2], Image)

        prediction = learner.infer(point_cloud, projected=False)
        self.assertIsInstance(prediction[0], np.ndarray)
        self.assertIsInstance(prediction[1], np.ndarray)
        self.assertIsNone(prediction[2])

    def test_infer_batch_point_clouds(self):
        # TODO: Verify Test files after uploading to server
        pcl_filenames = [
            self.test_data / "infer_data" / "seq08_f000000.bin",
            self.test_data / "infer_data" / "seq08_f000010.bin",
        ]
        point_clouds = [SemanticKittiDataset.load_point_cloud(f) for f in pcl_filenames]
        learner = EfficientLpsLearner()
        learner.load(self.model_weights)
        predictions = learner.infer(point_clouds, projected=True)
        for prediction in predictions:
            self.assertIsInstance(prediction[0], Heatmap)
            self.assertIsInstance(prediction[1], Heatmap)
            self.assertIsInstance(prediction[2], Image)

        predictions = learner.infer(point_clouds, projected=False)
        for prediction in predictions:
            self.assertIsInstance(prediction[0], np.ndarray)
            self.assertIsInstance(prediction[1], np.ndarray)
            self.assertIsNone(prediction[2])

    def test_save(self):
        # The model has not been trained.
        warnings.simplefilter("ignore", UserWarning)

        learner = EfficientLpsLearner()
        temp_model_path = self.temp_dir / "checkpoints"
        # Make sure that no model has been written to that path yet
        if temp_model_path.exists():
            rmdir(temp_model_path)
        successful = learner.save(temp_model_path)
        self.assertTrue((temp_model_path / "efficient_lps" / "efficient_lps.json").exists())
        self.assertTrue((temp_model_path / "efficient_lps" / "model.pth").exists())
        self.assertTrue(successful)
        rmdir(temp_model_path)

    def test_load_pretrained(self):
        learner = EfficientLpsLearner()
        successful = learner.load(self.model_weights)
        self.assertTrue(learner._is_model_trained)
        self.assertTrue(successful)

    def test_save_visualization(self):
        # TODO: Verify Test file after uploading to server
        point_cloud_filename = self.test_data / "infer_data" / "seq08_f000000.bin"
        temp_prediction_path = self.temp_dir / "prediction.png"
        point_cloud = SemanticKittiDataset.load_point_cloud(point_cloud_filename)
        learner = EfficientLpsLearner()
        learner.load(self.model_weights)
        prediction = learner.infer(point_cloud, projected=False)[:2]
        # Make sure that no file has been written to that path yet
        if temp_prediction_path.exists():
            rmfile(temp_prediction_path)
        EfficientLpsLearner.visualize(point_cloud, prediction, show_figure=False, save_figure=True,
                                     figure_filename=temp_prediction_path)
        self.assertTrue(temp_prediction_path.exists())
        rmfile(temp_prediction_path)

    def test_semantic_kitti_dataset(self):
        # Test the iterator methods of the dataset
        # TODO: comment out this test, since it depends on having the full Kitti Dataset installed somewhere
        kitti_path = "/home/arceyd/MasterThesis/dat/kitti/dataset/"

        for split in ["train", "valid", "test"]:
            dataset = SemanticKittiDataset(kitti_path, split=split)

            # Test the length of the dataset
            len_dataset = len(dataset)
            self.assertIsInstance(len_dataset, int)
            self.assertGreaterEqual(len_dataset, 0)

            # Get the 11th pointcloud from the dataset
            i10 = dataset[10]
            self.assertIsInstance(i10[0], PointCloud)
            self.assertIsNone(i10[1])


if __name__ == "__main__":
    unittest.main()
