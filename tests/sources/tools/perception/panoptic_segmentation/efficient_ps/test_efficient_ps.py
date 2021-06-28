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
import shutil
import unittest
import warnings
import zipfile
from typing import List, Tuple

import cv2

from opendr.engine.data import Image
from opendr.engine.target import Heatmap
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset
from opendr.perception.panoptic_segmentation.datasets import Image as ImageWithFilename
from opendr.perception.panoptic_segmentation.efficient_ps import EfficientPsLearner


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
    def setUpClass(cls) -> None:
        print('\n\n**********************************\nTEST EfficientPS Learner\n**********************************')

        cls.temp_dir = os.path.join('tests', 'sources', 'tools', 'perception', 'panoptic_segmentation', 'efficient_ps',
                                    'efficient_ps_temp')
        os.makedirs(cls.temp_dir)

        # Download all required files for testing
        cls.model_weights = EfficientPsLearner.download(path=cls.temp_dir, trained_on='cityscapes')
        test_data_zipped = EfficientPsLearner.download(path=cls.temp_dir, mode='test_data')
        cls.test_data = os.path.join(cls.temp_dir, 'test_data')
        with zipfile.ZipFile(test_data_zipped, 'r') as f:
            f.extractall(cls.test_data)

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up downloaded files
        rmdir(cls.temp_dir)

    def test_init(self):
        # Verify that the internal variables are initialized as expected by the other functions
        learner = EfficientPsLearner()
        self.assertFalse(learner._is_model_trained)

    def test_fit(self):
        pass

    def test_eval(self):
        # From mmdet base code
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', DeprecationWarning)

        val_dataset = CityscapesDataset(path=os.path.join(self.test_data, 'eval_data'))
        learner = EfficientPsLearner(batch_size=1)
        learner.load(self.model_weights)
        eval_results = learner.eval(val_dataset)
        self.assertIsInstance(eval_results, dict)

    def test_infer_single_image(self):
        image_filename = os.path.join(self.test_data, 'infer_data', 'lindau_000001_000019.png')
        image = Image(cv2.imread(image_filename))
        learner = EfficientPsLearner()
        learner.load(self.model_weights)
        prediction: Tuple[Heatmap, Heatmap] = learner.infer(image)
        for heatmap in prediction:
            self.assertIsInstance(heatmap, Heatmap)

        image_with_filename = ImageWithFilename(cv2.imread(image_filename), filename='lindau_000001_000019.png')
        prediction: Tuple[Heatmap, Heatmap] = learner.infer(image_with_filename)
        for heatmap in prediction:
            self.assertIsInstance(heatmap, Heatmap)

    def test_infer_batch_images(self):
        image_filenames = [
            os.path.join(self.test_data, 'infer_data', 'lindau_000001_000019.png'),
            os.path.join(self.test_data, 'infer_data', 'lindau_000003_000019.png'),
        ]
        images = [Image(cv2.imread(f)) for f in image_filenames]
        learner = EfficientPsLearner()
        learner.load(self.model_weights)
        predictions: List[Tuple[Heatmap, Heatmap]] = learner.infer(images)
        for prediction in predictions:
            for heatmap in prediction:
                self.assertIsInstance(heatmap, Heatmap)

    def test_save(self):
        # The model has not been trained.
        warnings.simplefilter('ignore', UserWarning)

        learner = EfficientPsLearner()
        temp_model_path = os.path.join(self.temp_dir, 'model.pth')
        # Make sure that no model has been written to that path yet
        if os.path.exists(temp_model_path):
            rmfile(temp_model_path)
        successful = learner.save(temp_model_path)
        self.assertTrue(os.path.exists(temp_model_path))
        self.assertTrue(successful)
        rmfile(temp_model_path)

    def test_load(self):
        learner = EfficientPsLearner()
        successful = learner.load(self.model_weights)
        self.assertTrue(learner._is_model_trained)
        self.assertTrue(successful)


if __name__ == '__main__':
    unittest.main()
