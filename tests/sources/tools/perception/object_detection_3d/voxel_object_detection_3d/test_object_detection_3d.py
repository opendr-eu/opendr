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

import unittest
import shutil
import os
import torch
from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import (
    VoxelObjectDetection3DLearner
)
from perception.object_detection_3d.datasets.kitti import KittiDataset
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.run import (
    example_convert_to_torch,
)
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.data.preprocess import (
    merge_second_batch,
)


DEVICE = "cpu"


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
        cls.temp_dir = os.path.join(".", "tests", "sources", "tools",
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
            ".", "src", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "pointpillars",
            "car", "test_short.proto")

        cls.config_pointpillars_ped_cycle = os.path.join(
            ".", "src", "perception", "object_detection_3d",
            "voxel_object_detection_3d", "second_detector", "configs", "pointpillars",
            "ped_cycle", "test_short.proto")

        cls.subsets_path = os.path.join(
            ".", "src", "perception", "object_detection_3d",
            "datasets", "mini_kitti_subsets")

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

        # cls.dataset_path = os.path.join("/", "data", "sets", "opendr_kitti")
        cls.dataset_path = os.path.join("/", "data", "sets", "opendr_mini_kitti")

        # Download all required files for testing
        # cls.pose_estimator.download(path=os.path.join(cls.temp_dir, "tanet_xyres_16_pretrained"))

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files

        for name in cls.all_configs.keys():
            rmdir(os.path.join(cls.temp_dir, "test_fit_" + name))
        rmdir(os.path.join(cls.temp_dir))

    def test_fit(self):

        def test_model(name, config):
            model_path = os.path.join(self.temp_dir, "test_fit_" + name)
            dataset = KittiDataset(self.dataset_path, self.subsets_path)

            learner = VoxelObjectDetection3DLearner(model_config_path=config, device=DEVICE)

            starting_param = list(learner.model.parameters())[0].clone()
            learner.fit(dataset,
                        auto_save=True,
                        model_dir=model_path)
            new_param = list(learner.model.parameters())[0].clone()
            self.assertFalse(torch.equal(starting_param, new_param))

        for name, config in self.car_configs.items():
            test_model(name, config)

    def test_eval(self):
        def test_model(name, config):
            model_path = os.path.join(self.temp_dir, "test_eval_" + name)
            dataset = KittiDataset(self.dataset_path, self.subsets_path)

            learner = VoxelObjectDetection3DLearner(model_config_path=config, device=DEVICE)
            learner.load(model_path)
            mAPbbox, mAPbev, mAP3d, mAPaos = learner.eval(dataset)

            # self.assertTrue(mAPbbox[0][0][0] > 80 and mAPbbox[0][0][0] < 95)
            self.assertTrue(mAPbbox[0][0][0] >= 0 and mAPbbox[0][0][0] < 95)

        for name, config in self.car_configs.items():
            test_model(name, config)

    def test_infer(self):
        def test_model(name, config):
            model_path = os.path.join(self.temp_dir, "test_eval_" + name)
            dataset = KittiDataset(self.dataset_path, self.subsets_path)

            learner = VoxelObjectDetection3DLearner(model_config_path=config, device=DEVICE)
            learner.load(model_path)

            (_, eval_dataset_iterator, ground_truth_annotations,) = learner._prepare_datasets(
                None,
                dataset,
                learner.input_config,
                learner.evaluation_input_config,
                learner.model_config,
                learner.voxel_generator,
                learner.target_assigner,
                None,
                require_dataset=False,
            )

            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset_iterator,
                batch_size=learner.evaluation_input_config.batch_size,
                shuffle=False,
                num_workers=learner.evaluation_input_config.num_workers,
                pin_memory=False,
                collate_fn=merge_second_batch,
            )

            result = learner.infer(
                example_convert_to_torch(next(iter(eval_dataloader)), learner.float_dtype, device=learner.device)
            )

            if len(result) == 2:
                self.assertTrue(len(result[1][0]["bbox"]) > 0)
            else:
                self.assertTrue(len(result[0]["bbox"]) > 0)

        for name, config in self.car_configs.items():
            test_model(name, config)

    # def test_save_load(self):
    #     self.pose_estimator.model = None
    #     self.pose_estimator.ort_session = None
    #     self.pose_estimator.init_model()
    #     self.pose_estimator.save(os.path.join(self.temp_dir, "testModel"))
    #     self.pose_estimator.model = None
    #     self.pose_estimator.load(os.path.join(self.temp_dir, "testModel"))
    #     self.assertIsNotNone(self.pose_estimator.model, "model is None after loading pth model.")
    #     # Cleanup
    #     rmdir(os.path.join(self.temp_dir, "testModel"))

    # def test_save_load_onnx(self):
    #     self.pose_estimator.model = None
    #     self.pose_estimator.ort_session = None
    #     self.pose_estimator.init_model()
    #     self.pose_estimator.optimize()
    #     self.pose_estimator.save(os.path.join(self.temp_dir, "testModel"))
    #     self.pose_estimator.model = None
    #     self.pose_estimator.load(os.path.join(self.temp_dir, "testModel"))
    #     self.assertIsNotNone(self.pose_estimator.ort_session, "ort_session is None after loading onnx model.")
    #     # Cleanup
    #     rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))
    #     rmdir(os.path.join(self.temp_dir, "testModel"))

    # def test_optimize(self):
    #     self.pose_estimator.model = None
    #     self.pose_estimator.ort_session = None
    #     self.pose_estimator.load(os.path.join(self.temp_dir, "trainedModel"))
    #     self.pose_estimator.optimize()
    #     self.assertIsNotNone(self.pose_estimator.ort_session)
    #     # Cleanup
    #     rmfile(os.path.join(self.temp_dir, "onnx_model_temp.onnx"))


if __name__ == "__main__":
    unittest.main()
