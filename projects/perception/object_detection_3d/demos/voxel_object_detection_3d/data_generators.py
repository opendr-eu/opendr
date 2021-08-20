from time import sleep
import numpy as np
from opendr.engine.data import PointCloud
from opendr.engine.datasets import PointCloudsDatasetIterator


def disk_point_cloud_generator(path, num_point_features=4, cycle=True, count=None):
    dataset = PointCloudsDatasetIterator(path, num_point_features=num_point_features)

    i = 0

    len_dataset = len(dataset) if count is None else count

    while i < len_dataset or cycle:
        yield dataset[i % len_dataset]
        i += 1


def lidar_point_cloud_generator(lidar):

    while True:
        yield lidar.next()
