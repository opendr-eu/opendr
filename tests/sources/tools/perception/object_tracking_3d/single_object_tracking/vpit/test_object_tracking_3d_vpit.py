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
import os
import torch
from opendr.perception.object_tracking_3d import ObjectTracking3DVpitLearner
from opendr.perception.object_tracking_3d import (
    LabeledTrackingPointCloudsDatasetIterator,
    SiameseTrackingDatasetIterator,
)

DEVICE = os.getenv("TEST_DEVICE") if os.getenv("TEST_DEVICE") else "cpu"

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


class TestObjectTracking3DVpit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(
            "\n\n**********************************\nTEST Object Tracking 3D VPIT Learner\n"
            "**********************************"
        )
        cls.temp_dir = os.path.join(
            "tests",
            "sources",
            "tools",
            "perception",
            "object_tracking_3d",
            "single_object_tracking",
            "vpit",
            "vpit_temp",
        )

        config_root = "./src/opendr/perception/object_tracking_3d/single_object_tracking/vpit/second_detector/configs"

        config_tanet_car = os.path.join(config_root, "tanet/car/xyres_16.proto")
        config_pointpillars_car = os.path.join(config_root, "pointpillars/car/xyres_16.proto")
        config_pointpillars_car_tracking = os.path.join(config_root, "pointpillars/car/xyres_16_tracking.proto")
        config_pointpillars_car_tracking_s = os.path.join(config_root, "pointpillars/car/xyres_16_tracking_s.proto")
        config_tanet_car_tracking = os.path.join(config_root, "tanet/car/xyres_16_tracking.proto")
        config_tanet_car_tracking_s = os.path.join(config_root, "tanet/car/xyres_16_tracking_s.proto")

        cls.backbone_configs = {
            "pp": config_pointpillars_car,
            "spp": config_pointpillars_car_tracking,
            "spps": config_pointpillars_car_tracking_s,
            "tanet": config_tanet_car,
            "stanet": config_tanet_car_tracking,
            "stanets": config_tanet_car_tracking_s,
        }

        cls.models_to_test = [("vpit", "pp")]

        cls.dataset = LabeledTrackingPointCloudsDatasetIterator.download_pico_kitti(
            cls.temp_dir, True
        )

        cls.dataset_siamese = SiameseTrackingDatasetIterator(
            [cls.dataset.lidar_path],
            [cls.dataset.label_path],
            [cls.dataset.calib_path],
            classes=["Van", "Pedestrian", "Cyclist"],
        )

        print("Dataset downloaded", file=sys.stderr)

    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files

        rmdir(os.path.join(cls.temp_dir))

    def test_fit(self):
        def test_model(name, backbone):
            print("Fit", name, "start", file=sys.stderr)
            model_path = os.path.join(self.temp_dir, "test_fit_" + name)

            learner = ObjectTracking3DVpitLearner(
                model_config_path=self.backbone_configs[backbone],
                device=DEVICE,
                checkpoint_after_iter=2,
            )

            starting_param = list(learner.model.parameters())[0].clone()
            learner.fit(
                self.dataset_siamese,
                model_dir=model_path,
                verbose=True,
                evaluate=False,
                steps=1,
            )
            new_param = list(learner.model.parameters())[0].clone()
            self.assertFalse(torch.equal(starting_param, new_param))

            print("Fit", name, "ok", file=sys.stderr)

        for name, backbone in self.models_to_test:
            test_model(name, backbone)

    def test_save(self):
        def test_model(name, backbone):
            print("Save", name, "start", file=sys.stderr)
            model_path = os.path.join(self.temp_dir, "test_save_" + name)
            save_path = os.path.join(model_path, "save")

            learner = ObjectTracking3DVpitLearner(
                model_config_path=self.backbone_configs[backbone],
                device=DEVICE,
                checkpoint_after_iter=2,
            )
            learner.save(save_path, True)
            starting_param_1 = list(learner.model.parameters())[0].clone()

            learner2 = ObjectTracking3DVpitLearner(
                model_config_path=self.backbone_configs[backbone],
                device=DEVICE,
                checkpoint_after_iter=2,
            )
            starting_param_2 = list(learner2.model.parameters())[0].clone()
            learner2.load(save_path)

            new_param = list(learner2.model.parameters())[0].clone()
            self.assertFalse(torch.equal(starting_param_1, starting_param_2))
            self.assertTrue(torch.equal(starting_param_1, new_param))

            print("Save", name, "ok", file=sys.stderr)

        for name, backbone in self.models_to_test:
            test_model(name, backbone)

    def test_unsupported(self):
        def test_model(name, backbone):
            learner = ObjectTracking3DVpitLearner(
                model_config_path=self.backbone_configs[backbone], device=DEVICE
            )

            with self.assertRaises(NotImplementedError):
                learner.optimize(None)

        for name, backbone in self.models_to_test:
            test_model(name, backbone)


if __name__ == "__main__":
    unittest.main()
