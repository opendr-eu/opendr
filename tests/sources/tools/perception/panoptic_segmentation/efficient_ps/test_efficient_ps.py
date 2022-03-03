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

from opendr.engine.data import Image
from opendr.engine.target import Heatmap
from opendr.perception.panoptic_segmentation import EfficientPsLearner, CityscapesDataset


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


class TestEfficientPsLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('\n\n**********************************\nTEST EfficientPS Learner\n**********************************')

        cls.temp_dir = os.path.join('tests', 'sources', 'tools', 'perception', 'panoptic_segmentation', 'efficient_ps',
                                    'efficient_ps_temp')
        if os.path.exists(cls.temp_dir):
            rmdir(cls.temp_dir)
        os.makedirs(cls.temp_dir)

        # Download all required files for testing
        cls.model_weights = EfficientPsLearner.download(path=cls.temp_dir, trained_on='cityscapes')
        test_data_zipped = EfficientPsLearner.download(path=cls.temp_dir, mode='test_data')
        cls.test_data = os.path.join(cls.temp_dir, 'test_data')
        with zipfile.ZipFile(test_data_zipped, 'r') as f:
            f.extractall(cls.test_data)

        # Configuration for the weights pre-trained on Cityscapes
        cls.config_file = str(Path(sys.modules[
                               EfficientPsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_cityscapes.py')

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        rmdir(cls.temp_dir)

    def test_init(self):
        # Verify that the internal variables are initialized as expected by the other functions

        learner = EfficientPsLearner(self.config_file)
        self.assertFalse(learner._is_model_trained)

    def test_fit(self):
        pass

    def test_eval(self):
        # From mmdet base code
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', DeprecationWarning)

        val_dataset = CityscapesDataset(path=os.path.join(self.test_data, 'eval_data'))
        learner = EfficientPsLearner(self.config_file, batch_size=1)
        learner.load(self.model_weights)
        eval_results = learner.eval(val_dataset)
        self.assertIsInstance(eval_results, dict)

    def test_infer_single_image(self):
        image_filename = os.path.join(self.test_data, 'infer_data', 'lindau_000001_000019.png')
        image = Image.open(image_filename)
        learner = EfficientPsLearner(self.config_file)
        learner.load(self.model_weights)
        prediction = learner.infer(image)
        for heatmap in prediction:
            self.assertIsInstance(heatmap, Heatmap)

    def test_infer_batch_images(self):
        image_filenames = [
            os.path.join(self.test_data, 'infer_data', 'lindau_000001_000019.png'),
            os.path.join(self.test_data, 'infer_data', 'lindau_000003_000019.png'),
        ]
        images = [Image.open(f) for f in image_filenames]
        learner = EfficientPsLearner(self.config_file)
        learner.load(self.model_weights)
        predictions = learner.infer(images)
        for prediction in predictions:
            for heatmap in prediction:
                self.assertIsInstance(heatmap, Heatmap)

    def test_save(self):
        # The model has not been trained.
        warnings.simplefilter('ignore', UserWarning)

        learner = EfficientPsLearner(self.config_file)
        temp_model_path = os.path.join(self.temp_dir, 'checkpoints')
        # Make sure that no model has been written to that path yet
        if os.path.exists(temp_model_path):
            rmdir(temp_model_path)
        successful = learner.save(temp_model_path)
        self.assertTrue(os.path.exists(os.path.join(temp_model_path, 'efficient_ps', 'efficient_ps.json')))
        self.assertTrue(os.path.exists(os.path.join(temp_model_path, 'efficient_ps', 'model.pth')))
        self.assertTrue(successful)
        rmdir(temp_model_path)

    def test_load_pretrained(self):
        learner = EfficientPsLearner(self.config_file)
        successful = learner.load(self.model_weights)
        self.assertTrue(learner._is_model_trained)
        self.assertTrue(successful)

    def test_save_visualization(self):
        image_filename = os.path.join(self.test_data, 'infer_data', 'lindau_000001_000019.png')
        temp_prediction_path = os.path.join(self.temp_dir, 'prediction.png')
        image = Image.open(image_filename)
        learner = EfficientPsLearner(self.config_file)
        learner.load(self.model_weights)
        prediction = learner.infer(image)
        # Make sure that no file has been written to that path yet
        if os.path.exists(temp_prediction_path):
            rmfile(temp_prediction_path)
        EfficientPsLearner.visualize(image, prediction, show_figure=False, save_figure=True,
                                     figure_filename=temp_prediction_path)
        self.assertTrue(os.path.exists(temp_prediction_path))
        rmfile(temp_prediction_path)
        EfficientPsLearner.visualize(image, prediction, show_figure=False, save_figure=True,
                                     figure_filename=temp_prediction_path, detailed=True)
        self.assertTrue(os.path.exists(temp_prediction_path))
        rmfile(temp_prediction_path)


if __name__ == '__main__':
    unittest.main()
