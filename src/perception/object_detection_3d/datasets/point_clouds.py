
import os
import numpy as np
from engine.datasets import DatasetIterator
from engine.data import PointCloud


class PointCloudsDatasetIterator(DatasetIterator):
    def __init__(self, path, num_point_features=4):
        super().__init__()

        self.path = path
        self.num_point_features = num_point_features
        self.files = os.listdir(path)

    def __getitem__(self, idx):
        data = np.fromfile(
            str(self.path + "/" + self.files[idx]), dtype=np.float32, count=-1
        ).reshape([-1, self.num_point_features])

        return PointCloud(data)

    def __len__(self):
        return len(self.files)
