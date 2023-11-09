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
import unittest
import shutil
import torch
from opendr.perception.object_tracking_2d import ObjectTracking2DDeepSortLearner
from opendr.perception.object_tracking_2d import (
    Market1501Dataset,
    Market1501DatasetIterator,
)
from opendr.perception.object_tracking_2d import (
    MotDataset,
    RawMotWithDetectionsDatasetIterator,
)
import os

DEVICE = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'

print("Using device:", DEVICE)
print("Using device:", DEVICE, file=sys.stderr)


def rmfile(path):
    try:
        os.remove(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def rmdir(_dir):
    try:
        shutil.rmtree(_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


class TestObjectTracking2DDeepSortLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join("tests", "sources", "tools",
                                    "perception", "object_tracking_2d",
                                    "deep_sort",
                                    "deep_sort_temp")

        cls.train_split_paths = {
            "nano_mot20": os.path.join(
                ".", "src", "opendr", "perception", "object_tracking_2d",
                "datasets", "splits", "nano_mot20.train"
            )
        }

        cls.model_names = [
            "deep_sort",
        ]

        cls.mot_dataset_path = MotDataset.download_nano_mot20(
            os.path.join(cls.temp_dir, "mot_dataset"), True
        ).path
        cls.market1501_dataset_path = Market1501Dataset.download_nano_market1501(
            os.path.join(cls.temp_dir, "market1501_dataset"), True
        ).path

        print("Dataset downloaded", file=sys.stderr)

        for model_name in cls.model_names:
            ObjectTracking2DDeepSortLearner.download(
                model_name, cls.temp_dir
            )

        print("Models downloaded", file=sys.stderr)

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files

        rmdir(os.path.join(cls.temp_dir))

    def test_fit(self):

        def test_model(name):
            dataset = Market1501Dataset(self.market1501_dataset_path)

            learner = ObjectTracking2DDeepSortLearner(
                temp_path=self.temp_dir,
                device=DEVICE,
            )

            starting_param = list(learner.tracker.deepsort.extractor.net.parameters())[0].clone()

            learner.fit(
                dataset,
                epochs=2,
                val_epochs=2,
                verbose=True,
            )
            new_param = list(learner.tracker.deepsort.extractor.net.parameters())[0].clone()
            self.assertFalse(torch.equal(starting_param, new_param))

            print("Fit", name, "ok", file=sys.stderr)

        for name in self.model_names:
            test_model(name)

    def test_fit_iterator(self):
        def test_model(name):
            dataset = Market1501DatasetIterator(
                os.path.join(self.market1501_dataset_path, "bounding_box_train"),
            )
            eval_dataset = Market1501DatasetIterator(
                os.path.join(self.market1501_dataset_path, "bounding_box_test"),
            )

            learner = ObjectTracking2DDeepSortLearner(
                checkpoint_after_iter=3,
                temp_path=self.temp_dir,
                device=DEVICE,
            )

            starting_param = list(learner.tracker.deepsort.extractor.net.parameters())[0].clone()

            learner.fit(
                dataset,
                epochs=2,
                val_dataset=eval_dataset,
                val_epochs=2,
                verbose=True,
            )
            new_param = list(learner.tracker.deepsort.extractor.net.parameters())[0].clone()
            self.assertFalse(torch.equal(starting_param, new_param))

            print("Fit iterator", name, "ok", file=sys.stderr)

        for name in self.model_names:
            test_model(name)

    def test_eval(self):
        def test_model(name):
            model_path = os.path.join(self.temp_dir, name)
            train_split_paths = {
                "nano_mot20": os.path.join(
                    ".", "src", "opendr", "perception", "object_tracking_2d",
                    "datasets", "splits", "nano_mot20.train"
                )
            }

            dataset = RawMotWithDetectionsDatasetIterator(
                self.mot_dataset_path,
                train_split_paths
            )

            learner = ObjectTracking2DDeepSortLearner(
                temp_path=self.temp_dir,
                device=DEVICE,
            )
            learner.load(model_path, verbose=True)
            result = learner.eval(dataset)

            self.assertGreater(len(result["mota"]), 0)

        for name in self.model_names:
            test_model(name)

    def test_infer(self):
        def test_model(name):
            model_path = os.path.join(self.temp_dir, name)
            train_split_paths = {
                "nano_mot20": os.path.join(
                    ".", "src", "opendr", "perception", "object_tracking_2d",
                    "datasets", "splits", "nano_mot20.train"
                )
            }

            dataset = RawMotWithDetectionsDatasetIterator(
                self.mot_dataset_path,
                train_split_paths
            )

            learner = ObjectTracking2DDeepSortLearner(
                temp_path=self.temp_dir,
                device=DEVICE,
            )
            learner.load(model_path, verbose=True)
            result = learner.infer(dataset[0][0], 1)

            self.assertTrue(len(result) > 0)

            learner.reset()

            result = learner.infer([
                dataset[0][0],
                dataset[1][0],
            ])

            self.assertTrue(len(result) == 2)
            self.assertTrue(len(result[0]) > 0)

        for name in self.model_names:
            test_model(name)

    def test_save(self):
        def test_model(name):
            model_path = os.path.join(self.temp_dir, "test_save_" + name)
            save_path = os.path.join(model_path, "save")

            learner = ObjectTracking2DDeepSortLearner(
                temp_path=self.temp_dir,
                device=DEVICE,
            )

            learner.save(save_path, True)
            starting_param_1 = list(learner.tracker.deepsort.extractor.net.parameters())[0].clone()

            learner2 = ObjectTracking2DDeepSortLearner(
                temp_path=self.temp_dir,
                device=DEVICE,
            )
            learner2.load(save_path)

            new_param = list(learner2.tracker.deepsort.extractor.net.parameters())[0].clone()
            self.assertTrue(torch.equal(starting_param_1, new_param))

        for name in self.model_names:
            test_model(name)

    def test_optimize(self):
        def test_model(name):
            model_path = os.path.join(self.temp_dir, name)
            train_split_paths = {
                "nano_mot20": os.path.join(
                    ".", "src", "opendr", "perception", "object_tracking_2d",
                    "datasets", "splits", "nano_mot20.train"
                )
            }

            dataset = RawMotWithDetectionsDatasetIterator(
                self.mot_dataset_path,
                train_split_paths
            )

            learner = ObjectTracking2DDeepSortLearner(
                temp_path=self.temp_dir,
                device=DEVICE,
            )
            learner.load(model_path, verbose=True)
            learner.optimize()
            result = learner.eval(dataset)

            self.assertGreater(len(result["mota"]), 0)

        for name in self.model_names:
            test_model(name)


if __name__ == "__main__":
    unittest.main()
