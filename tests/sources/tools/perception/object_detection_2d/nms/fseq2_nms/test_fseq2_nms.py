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
import gc
import shutil
import os
import numpy as np
from opendr.perception.object_detection_2d import FSeq2NMSLearner
from opendr.perception.object_detection_2d.nms.utils.nms_dataset import Dataset_NMS
from opendr.engine.data import Image
from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner


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


class TestFSeq2NMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST FSeq2-NMS Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "object_detection_2d",
                                    "nms", "fseq2_nms", "temp")
        cls.fSeq2NMSLearner = FSeq2NMSLearner(iou_filtering=None, temp_path=cls.temp_dir,
                                              device='cpu',  checkpoint_after_iter=1, epochs=1)

        # Download all required files for testing
        cls.fseq2_nms_model = 'fseq2_pets_ssd_pets'
        cls.fSeq2NMSLearner.download(model_name=cls.fseq2_nms_model, path=cls.temp_dir)
        cls.ssd_model = 'ssd_default_person'

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for FSeq2-NMS...')
        # Clean up downloaded files
        rmfile(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "test_module.pkl"))
        rmfile(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "val2014", "COCO_val2014_000000262148.jpg"))
        rmfile(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "annotations", "test_module_anns.json"))
        rmdir(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "val2014"))
        rmdir(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "FMoD"))
        rmdir(os.path.join(cls.temp_dir))
        del cls.fSeq2NMSLearner
        gc.collect()
        print('Finished cleaning for FSeq2-NMS...')

    def test_fit(cls):
        print('Starting training test for FSeq2-NMS...')

        m = list(cls.fSeq2NMSLearner.model.parameters())[0].clone()
        cls.fSeq2NMSLearner.fit(dataset='TEST_MODULE', datasets_folder=cls.temp_dir + '/datasets', logging_path=None,
                                silent=False, verbose=True, nms_gt_iou=0.50, max_dt_boxes=200, ssd_model=cls.ssd_model)
        n = list(cls.fSeq2NMSLearner.model.parameters())[0].clone()
        cls.assertFalse(np.array_equal(m, n), msg="Model parameters did not change after running fit.")
        del m, n
        gc.collect()
        print('Finished training test for FSeq2-NMS...')

    def test_eval(cls):
        print('Starting evaluation test for FSeq2-NMS...')
        cls.fSeq2NMSLearner.load(os.path.join(cls.temp_dir, cls.fseq2_nms_model), verbose=True)
        results_dict = cls.fSeq2NMSLearner.eval(dataset='TEST_MODULE', split='test', max_dt_boxes=800,
                                                datasets_folder=cls.temp_dir + '/datasets', ssd_model=cls.ssd_model)
        if results_dict is None:
            cls.assertIsNotNone(results_dict, msg="Eval results dictionary not returned.")
        else:
            cls.assertGreater(results_dict[0][0][1][0], 0.4)
        del results_dict
        gc.collect()
        print('Finished evaluation test for FSeq2-NMS...')

    def test_infer(cls):
        print('Starting inference test for FSeq2-NMS...')
        cls.fSeq2NMSLearner.load(os.path.join(cls.temp_dir, cls.fseq2_nms_model), verbose=True)
        dataset_nms = Dataset_NMS(path=cls.temp_dir + '/datasets', dataset_name='TEST_MODULE', split='train',
                                  use_ssd=True, use_maps=True, ssd_model=cls.ssd_model)
        image_fln = dataset_nms.src_data[0]['filename']
        img = Image.open(os.path.join(cls.temp_dir, 'datasets', 'TEST_MODULE', image_fln))

        ssd = SingleShotDetectorLearner(device='cpu')
        ssd.download(cls.temp_dir, mode="pretrained")
        ssd.load(os.path.join(cls.temp_dir, cls.ssd_model), verbose=True)
        bounding_box_list, _ = ssd.infer(img, threshold=0.2, custom_nms=cls.fSeq2NMSLearner)

        cls.assertIsNotNone(bounding_box_list, msg="Returned empty BoundingBoxList.")
        del img
        del bounding_box_list
        del dataset_nms
        gc.collect()
        print('Finished inference test for FSeq2-NMS...')

    def test_save_load(self):
        print('Starting save/load test for FSeq2-NMS...')
        self.fSeq2NMSLearner.save(os.path.join(self.temp_dir, "test_model", "last_weights"), current_epoch=0)
        self.fSeq2NMSLearner.model = None
        self.fSeq2NMSLearner.init_model()
        self.fSeq2NMSLearner.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.fSeq2NMSLearner.model, "model is None after loading model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_model"))
        print('Finished save/load test for FSeq2-NMS...')


if __name__ == "__main__":
    unittest.main()
