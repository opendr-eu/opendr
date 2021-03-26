# Copyright 2020-2021 OpenDR Project
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
# import shutil
from perception.activity_recognition.x3d.x3d_learner import X3DLearner

# from engine.datasets import ExternalDataset
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)


class TestX3DLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(
            "./tests/sources/tools/perception/activity_recognition/x3d/temp"
        )

        # Download all required files for testing
        X3DLearner.download(path=Path(cls.temp_dir) / "weights", model_weights={"xs"})
        cls.x3d_learner = X3DLearner(
            device="cpu", temp_path=str(cls.temp_dir), batch_size=1, backbone="xs",
        )

    # @classmethod
    # def tearDownClass(cls):
    #     try:
    #         shutil.rmtree(str(cls.tmp_dir))
    #     except OSError as e:
    #         logger.error(f"Caught error while cleaning up {e.filename}: {e.strerror}")

    def test_downloaded(self):
        assert Path(self.temp_dir) / "weights" / "x3d_s.pyth"

    def test_save_and_load(self):
        assert self.x3d_learner.model is not None
        self.x3d_learner.save(self.temp_dir)
        # Make changes to check subsequent load
        self.x3d_learner.model = None
        self.x3d_learner.batch_size = 42
        self.x3d_learner.load(self.temp_dir)
        self.assertIsNotNone(
            self.x3d_learner.model, "model is None after loading pth model."
        )
        assert self.x3d_learner.batch_size == 1

    # def test_fit(self):
    #     training_dataset = ExternalDataset(
    #         path=os.path.join(self.temp_dir, "dataset"), dataset_type="COCO"
    #     )
    #     self.pose_estimator.model = None
    #     self.pose_estimator.init_model()
    #     m = list(self.pose_estimator.model.parameters())[0].clone()
    #     self.pose_estimator.fit(
    #         dataset=training_dataset,
    #         silent=True,
    #         images_folder_name="image",
    #         annotations_filename="annotation.json",
    #     )
    #     self.assertFalse(
    #         torch.equal(m, list(self.pose_estimator.model.parameters())[0]),
    #         msg="Model parameters did not change after running fit.",
    #     )

    # def test_eval(self):
    #     eval_dataset = ExternalDataset(
    #         path=os.path.join(self.temp_dir, "dataset"), dataset_type="COCO"
    #     )
    #     self.pose_estimator.load(os.path.join(self.temp_dir, "trainedModel"))
    #     results_dict = self.pose_estimator.eval(
    #         eval_dataset,
    #         use_subset=False,
    #         verbose=True,
    #         silent=True,
    #         images_folder_name="image",
    #         annotations_filename="annotation.json",
    #     )
    #     self.assertNotEqual(
    #         len(results_dict["average_precision"]),
    #         0,
    #         msg="Eval results dictionary contains empty list.",
    #     )
    #     self.assertNotEqual(
    #         len(results_dict["average_recall"]),
    #         0,
    #         msg="Eval results dictionary contains empty list.",
    #     )
    #     # Cleanup
    #     rmfile(os.path.join(self.temp_dir, "detections.json"))

    # def test_infer(self):
    #     self.pose_estimator.model = None
    #     self.pose_estimator.load(os.path.join(self.temp_dir, "trainedModel"))

    #     img = cv2.imread(
    #         os.path.join(self.temp_dir, "dataset", "image", "000000000785.jpg")
    #     )
    #     # Default pretrained mobilenet model detects 18 keypoints on img with id 785
    #     self.assertGreater(
    #         len(self.pose_estimator.infer(img)[0].data),
    #         0,
    #         msg="Returned pose must have non-zero number of keypoints.",
    #     )

    # def test_save_load_onnx(self):
    #     self.pose_estimator.model = None
    #     self.pose_estimator.ort_session = None
    #     self.pose_estimator.init_model()
    #     self.pose_estimator.optimize()
    #     self.pose_estimator.save(os.path.join(self.temp_dir, "testModel"))
    #     self.pose_estimator.model = None
    #     self.pose_estimator.load(os.path.join(self.temp_dir, "testModel"))
    #     self.assertIsNotNone(
    #         self.pose_estimator.ort_session,
    #         "ort_session is None after loading onnx model.",
    #     )
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
