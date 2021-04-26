import unittest
import torch
import numpy as np

from engine.target import Category


class TestTarget(unittest.TestCase):

    def test_category(self):
        data_list = [0.1, 0.5, 0.2, 0.2]
        # torch.Tensor
        c_t = Category(torch.tensor(data_list))
        assert c_t.num_classes == 4
        assert torch.allclose(c_t.data, torch.tensor(data_list))
        assert c_t.prediction == 1

        # np.ndarray
        c_t = Category(np.array(data_list))
        assert c_t.num_classes == 4
        assert torch.allclose(c_t.data, torch.tensor(data_list))
        assert c_t.prediction == 1

        # int, one hot
        c_int = Category(data=1, num_classes=4)
        assert c_int.num_classes == 4
        assert torch.allclose(c_int.data, torch.tensor([0.0, 1.0, 0.0, 0.0]))
        assert c_int.prediction == 1


if __name__ == "__main__":
    unittest.main()
