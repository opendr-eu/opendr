import unittest
import torch
import numpy as np

from opendr.engine.target import Category


class TestTarget(unittest.TestCase):

    def test_category(self):
        data_list = [0.1, 0.5, 0.2, 0.2]
        # torch.Tensor
        c_t = Category(prediction=1, confidence=torch.tensor(data_list))
        assert c_t.data == 1
        assert torch.allclose(c_t.confidence, torch.tensor(data_list))

        # np.ndarray
        c_t = Category(prediction=1, confidence=np.array(data_list))


if __name__ == "__main__":
    unittest.main()
