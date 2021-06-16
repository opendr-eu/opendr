# Copyright 2020-2021 OpenDR European Project
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
import os
import torch
from opendr.engine.datasets import PointCloudsDatasetIterator
from opendr.perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
)
from opendr.perception.object_detection_3d.datasets.kitti import KittiDataset, LabeledPointCloudsDatasetIterator


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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


class TestVoxelObjectDetection3DLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n****************************************\nTEST Voxel Object Detection 3D Learner\n"
              "****************************************")
        cls.temp_dir = os.path.join("tests", "sources", "tools",
                                    "perception", "object_detection_3d",
                                    "voxel_object_detection_3d",
                                    "voxel_object_detection_3d_temp")

        cls.config_tanet_car = os.path.join(".", "src", "perception",
                                            "object_detection_3d",
                                            "voxel_object_detection_3d",
                                            "second_detector", "configs", "tanet",
                                            "car", "test_short.proto")

        cls.config_tanet_ped_cycle = os.path.join(".", "src", "perception",
                                                  "object_detection_3d",
                                                  "voxel_object_detection_3d",
                                                  "second_detector", "configs", "tanet",
                                                  "ped_cycle",
                                                  "test_short.proto")

        cls.config_pointpillars_car = os.path.join(
            ".", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "pointpillars",
            "car", "test_short.proto")

        cls.config_pointpillars_ped_cycle = os.path.join(
            ".", "src", "opendr", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "pointpillars",
            "ped_cycle", "test_short.proto")

        cls.subsets_path = os.path.join(
            ".", "src", "opendr", "perception", "object_detection_3d",
            "datasets", "nano_kitti_subsets")

        cls.download_model_names = {
            "tanet_car": "tanet_car_xyres_16",
            "tanet_ped_cycle": "tanet_ped_cycle_xyres_16",
            "pointpillars_car": "pointpillars_car_xyres_16",
            "pointpillars_ped_cycle": "pointpillars_ped_cycle_xyres_16",
        }

        cls.all_configs = {
            "tanet_car": cls.config_tanet_car,
            "tanet_ped_cycle": cls.config_tanet_ped_cycle,
            "pointpillars_car": cls.config_pointpillars_car,
            "pointpillars_ped_cycle": cls.config_pointpillars_ped_cycle,
        }
        cls.car_configs = {
            "tanet_car": cls.config_tanet_car,
            "pointpillars_car": cls.config_pointpillars_car,
        }

        cls.dataset_path = KittiDataset.download_nano_kitti(
            cls.temp_dir, True, cls.subsets_path
        ).path

        print("Dataset downloaded", file=sys.stderr)

        for model_name in cls.download_model_names.values():
            VoxelObjectDetection3DLearner.download(
                model_name, cls.temp_dir
            )

        print("Models downloaded", file=sys.stderr)

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files

        rmdir(os.path.join(cls.temp_dir))
        pass

    def test_fit(self):

        def test_model(name, config):
            model_path = os.path.join(self.temp_dir, "test_fit_" + name)
            dataset = KittiDataset(self.dataset_path, self.subsets_path)

            learner = VoxelObjectDetection3DLearner(
                model_config_path=config, device=DEVICE,
                checkpoint_after_iter=2,
            )

            starting_param = list(learner.model.parameters())[0].clone()
            learner.fit(
                dataset,
                model_dir=model_path,
                verbose=True,
                evaluate=False,
            )
            new_param = list(learner.model.parameters())[0].clone()
            self.assertFalse(torch.equal(starting_param, new_param))

            print("Fit", name, "ok", file=sys.stderr)

        for name, config in self.car_configs.items():
            test_model(name, config)

    def test_fit_iterator(self):
        def test_model(name, config):
            model_path = os.path.join(self.temp_dir, "test_fit_iterator_" + name)
            dataset = LabeledPointCloudsDatasetIterator(
                self.dataset_path + "/training/velodyne_reduced",
                self.dataset_path + "/training/label_2",
                self.dataset_path + "/training/calib",
            )

            val_dataset = LabeledPointCloudsDatasetIterator(
                self.dataset_path + "/training/velodyne_reduced",
                self.dataset_path + "/training/label_2",
                self.dataset_path + "/training/calib",
            )

            learner = VoxelObjectDetection3DLearner(
                model_config_path=config, device=DEVICE,
                checkpoint_after_iter=90,
            )

            starting_param = list(learner.model.parameters())[0].clone()
            learner.fit(
                dataset,
                val_dataset=val_dataset,
                model_dir=model_path,
                evaluate=False,
            )
            new_param = list(learner.model.parameters())[0].clone()
            self.assertFalse(torch.equal(starting_param, new_param))

            print("Fit iterator", name, "ok", file=sys.stderr)

        for name, config in self.car_configs.items():
            test_model(name, config)

    def test_eval(self):
        def test_model(name, config):
            model_path = os.path.join(self.temp_dir, self.download_model_names[name])
            dataset = KittiDataset(self.dataset_path, self.subsets_path)

            learner = VoxelObjectDetection3DLearner(model_config_path=config, device=DEVICE)
            learner.load(model_path)
            mAPbbox, mAPbev, mAP3d, mAPaos = learner.eval(dataset, count=2)

            self.assertTrue(mAPbbox[0][0][0] > 1 and mAPbbox[0][0][0] < 95, msg=mAPbbox[0][0][0])

        for name, config in self.car_configs.items():
            test_model(name, config)

    def test_infer(self):
        def test_model(name, config):

            dataset = PointCloudsDatasetIterator(self.dataset_path + "/testing/velodyne_reduced")

            learner = VoxelObjectDetection3DLearner(
                model_config_path=config, device=DEVICE
            )

            result = learner.infer(
                dataset[0]
            )

            self.assertTrue(len(result) > 0)

            result = learner.infer(
                [dataset[0], dataset[1], dataset[2]]
            )
            self.assertTrue(len(result) == 3)
            self.assertTrue(len(result[0]) > 0)

        for name, config in self.car_configs.items():
            test_model(name, config)

    def test_save(self):
        def test_model(name, config):
            model_path = os.path.join(self.temp_dir, "test_save_" + name)
            save_path = os.path.join(model_path, "save")

            learner = VoxelObjectDetection3DLearner(
                model_config_path=config, device=DEVICE
            )
            learner.save(save_path, True)
            starting_param_1 = list(learner.model.parameters())[0].clone()

            learner2 = VoxelObjectDetection3DLearner(
                model_config_path=config, device=DEVICE
            )
            starting_param_2 = list(learner2.model.parameters())[0].clone()
            learner2.load(save_path)

            new_param = list(learner2.model.parameters())[0].clone()
            self.assertFalse(torch.equal(starting_param_1, starting_param_2))
            self.assertTrue(torch.equal(starting_param_1, new_param))

        for name, config in self.car_configs.items():
            test_model(name, config)

    def test_optimize(self):
        def test_model(name, config):
            model_path = os.path.join(self.temp_dir, "test_optimize_" + name)

            dataset = PointCloudsDatasetIterator(self.dataset_path + "/testing/velodyne_reduced")

            learner = VoxelObjectDetection3DLearner(
                model_config_path=config, device=DEVICE,
                temp_path=self.temp_dir
            )
            learner.optimize()

            result = learner.infer(
                dataset[0]
            )
            self.assertTrue(len(result) > 0)

            learner.save(model_path)

            learner2 = VoxelObjectDetection3DLearner(
                model_config_path=config, device=DEVICE
            )
            learner2.load(model_path, True)

            self.assertTrue(learner2.model.rpn_ort_session is not None)

        for name, config in self.car_configs.items():
            test_model(name, config)


if __name__ == "__main__":
    unittest.main()
