import json
import os
import shutil
import unittest

import numpy as np
import torch as t

from perception.speech_recognition.edgespeechnets.edgespeechnets_learner import EdgesSpeechNetsLearner
from engine.datasets import DatasetIterator

TEST_BATCH_SIZE = 2
TEST_EPOCHS = 1
TEST_CLASSES_N = 2
TEST_INFER_LENGTH = 2
TEST_SIGNAL_LENGTH = 16000

TEMP_SAVE_DIR = os.path.join(".", "tests", "sources", "tools", "perception", "speech_recognition",
                             "edgespeechnets", "edgespeechnets_temp")


class DummyDataset(DatasetIterator):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return TEST_BATCH_SIZE * 2

    def __getitem__(self, item):
        return np.ones(TEST_SIGNAL_LENGTH), np.random.choice(TEST_CLASSES_N)


class EdgeSpeechNetsTest(unittest.TestCase):
    learner = None

    @classmethod
    def setUpClass(cls):
        cls.learner = EdgesSpeechNetsLearner(device="cpu", output_classes_n=TEST_CLASSES_N, iters=TEST_EPOCHS)

    @classmethod
    def tearDownClass(cls):
        del cls.learner

    def test_fit(self):
        train_dataset = DummyDataset()
        test_dataset = DummyDataset()
        weights_before_fit = list(self.learner.model.parameters())[0].clone()
        results = self.learner.fit(dataset=train_dataset, val_dataset=test_dataset)
        self.assertFalse(t.equal(weights_before_fit, list(self.learner.model.parameters())[0]),
                         msg="Fit method did not alter model weights")
        self.assertTrue(len(results) == TEST_EPOCHS, f"The results dictionary length is not {TEST_EPOCHS}")

    def test_eval(self):
        eval_dataset = DummyDataset()
        results = self.learner.eval(dataset=eval_dataset)
        self.assertTrue(0.0 <= results["test_accuracy"] <= 1.0, "Test accuracy not between 0 and 1.")
        self.assertTrue(0.0 <= results["test_total_loss"], "Test total loss is negative")

    def test_infer_batch(self):
        batch = np.ones((TEST_INFER_LENGTH, TEST_SIGNAL_LENGTH))
        results = self.learner.infer(batch)
        self.assertTrue(results.shape == (TEST_INFER_LENGTH,))

    def test_infer_pure_signal(self):
        signal = np.ones(TEST_SIGNAL_LENGTH)
        results = self.learner.infer(signal)
        self.assertTrue(results.shape == (1,))

    def test_reset(self):
        weights_before_reset = list(self.learner.model.parameters())[0].clone()
        self.learner.reset()
        self.assertFalse(t.equal(weights_before_reset, list(self.learner.model.parameters())[0]),
                         msg="Reset method did not alter model weights")

    def test_save_load(self):
        weights_before_saving = list(self.learner.model.parameters())[0].clone()
        self.learner.save(TEMP_SAVE_DIR)
        self.learner.reset()
        self.assertFalse(t.equal(weights_before_saving, list(self.learner.model.parameters())[0]),
                         msg="Reset method did not alter model weights")
        self.learner.load(TEMP_SAVE_DIR)
        self.assertTrue(t.equal(weights_before_saving, list(self.learner.model.parameters())[0]),
                        msg="Load did not restore weights properly")
        with open(os.path.join(TEMP_SAVE_DIR, os.path.basename(TEMP_SAVE_DIR) + ".json")) as jsonfile:
            metadata = json.load(jsonfile)
        self.assertTrue(all(key in metadata for key in ["model_paths",
                                                        "framework",
                                                        "format",
                                                        "has_data",
                                                        "inference_params",
                                                        "optimized",
                                                        "optimizer_info"]))

        # Remove temporary files
        try:
            shutil.rmtree(TEMP_SAVE_DIR)
        except OSError as e:
            print(f"Exception when trying to remove temp directory: {e.strerror}")


if __name__ == "__main__":
    unittest.main()
