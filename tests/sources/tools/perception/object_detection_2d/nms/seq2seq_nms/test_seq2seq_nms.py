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
from opendr.perception.object_detection_2d import Seq2SeqNMSLearner
from opendr.perception.object_detection_2d.nms.utils.nms_dataset import Dataset_NMS
from opendr.engine.data import Image


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


class TestSeq2SeqNMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Seq2Seq-NMS Learner\n"
              "**********************************")

        cls.temp_dir = os.path.join(".", "tests", "sources", "tools", "perception", "object_detection_2d",
                                    "nms", "seq2seq_nms", "temp")
        cls.seq2SeqNMSLearner = Seq2SeqNMSLearner(iou_filtering=None, app_feats='fmod', temp_path=cls.temp_dir,
                                                  device='cpu',  checkpoint_after_iter=1, epochs=1)

        # Download all required files for testing
        cls.seq2SeqNMSLearner.download(model_name='seq2seq_pets_jpd_fmod', path=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        print('Removing temporary directories for Seq2Seq-NMS...')
        # Clean up downloaded files
        rmfile(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "test_module.pkl"))
        rmfile(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "val2014", "COCO_val2014_000000262148.jpg"))
        rmfile(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "FMoD", "coco_edgemap_b_3.pkl"))
        rmfile(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "annotations", "test_module_anns.json"))
        rmdir(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "val2014"))
        rmdir(os.path.join(cls.temp_dir, "datasets", "TEST_MODULE", "FMoD"))
        rmfile(os.path.join(cls.temp_dir, "seq2seq_pets_jpd_fmod", "fmod_normalization.pkl"))
        rmfile(os.path.join(cls.temp_dir, "seq2seq_pets_jpd_fmod", "last_weights.json"))
        rmfile(os.path.join(cls.temp_dir, "seq2seq_pets_jpd_fmod", "last_weights.pth"))
        rmdir(os.path.join(cls.temp_dir, "seq2seq_pets_jpd_fmod"))

        rmdir(os.path.join(cls.temp_dir))

        del cls.seq2SeqNMSLearner
        gc.collect()
        print('Finished cleaning for Seq2Seq-NMS...')

    def test_fit(self):
        print('Starting training test for Seq2Seq-NMS...')

        m = list(self.seq2SeqNMSLearner.model.parameters())[0].clone()
        self.seq2SeqNMSLearner.fit(dataset='TEST_MODULE', use_ssd=False,
                                   datasets_folder=self.temp_dir + '/datasets',
                                   logging_path=None, silent=False, verbose=True, nms_gt_iou=0.50,
                                   max_dt_boxes=200)
        n = list(self.seq2SeqNMSLearner.model.parameters())[0].clone()
        self.assertFalse(np.array_equal(m, n),
                         msg="Model parameters did not change after running fit.")
        del m, n
        gc.collect()
        print('Finished training test for Seq2Seq-NMS...')

    def test_eval(self):
        print('Starting evaluation test for Seq2Seq-NMS...')
        self.seq2SeqNMSLearner.load(self.temp_dir + '/seq2seq_pets_jpd_fmod/', verbose=True)
        results_dict = self.seq2SeqNMSLearner.eval(dataset='TEST_MODULE', split='test', max_dt_boxes=800,
                                                   datasets_folder=self.temp_dir + '/datasets',
                                                   use_ssd=False)
        if results_dict is None:
            self.assertIsNotNone(results_dict,
                                 msg="Eval results dictionary not returned.")
        else:
            self.assertGreater(results_dict[0][0][1][0], 0.4)
        del results_dict
        gc.collect()
        print('Finished evaluation test for Seq2Seq-NMS...')

    def test_infer(self):
        print('Starting inference test for Seq2Seq-NMS...')
        self.seq2SeqNMSLearner.load(self.temp_dir + '/seq2seq_pets_jpd_fmod/', verbose=True)
        dataset_nms = Dataset_NMS(path=self.temp_dir + '/datasets', dataset_name='TEST_MODULE', split='train', use_ssd=False)
        image_fln = dataset_nms.src_data[0]['filename']
        img = Image.open(os.path.join(self.temp_dir, 'datasets', 'TEST_MODULE', image_fln))
        boxes = dataset_nms.src_data[0]['dt_boxes'][1][:, 0:4]
        scores = np.expand_dims(dataset_nms.src_data[0]['dt_boxes'][1][:, 4], axis=-1)

        bounding_box_list = self.seq2SeqNMSLearner.run_nms(boxes=boxes, scores=scores, img=img, threshold=0.5)

        self.assertIsNotNone(bounding_box_list,
                             msg="Returned empty BoundingBoxList.")
        del img
        del bounding_box_list
        del boxes
        del scores
        del dataset_nms
        gc.collect()
        print('Finished inference test for Seq2Seq-NMS...')

    def test_save_load(self):
        print('Starting save/load test for Seq2Seq-NMS...')
        self.seq2SeqNMSLearner.save(os.path.join(self.temp_dir, "test_model", "last_weights"), current_epoch=0)
        self.seq2SeqNMSLearner.model = None
        self.seq2SeqNMSLearner.init_model()
        self.seq2SeqNMSLearner.load(os.path.join(self.temp_dir, "test_model"))
        self.assertIsNotNone(self.seq2SeqNMSLearner.model, "model is None after loading model.")
        # Cleanup
        rmdir(os.path.join(self.temp_dir, "test_model"))
        print('Finished save/load test for Seq2Seq-NMS...')


if __name__ == "__main__":
    unittest.main()
