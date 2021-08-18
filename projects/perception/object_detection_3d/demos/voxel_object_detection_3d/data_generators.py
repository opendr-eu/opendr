from time import sleep
import numpy as np
from opendr.engine.data import PointCloud
from opendr.engine.datasets import PointCloudsDatasetIterator


def disk_point_cloud_generator(path, num_point_features=4, cycle=True):
    dataset = PointCloudsDatasetIterator(path, num_point_features=num_point_features)

    i = 0

    while i < len(dataset) or cycle:
        yield dataset[i % len(dataset)]
        i += 1
        sleep(2.0)


def lidar_point_cloud_generator(lidar):

    while True:
        yield lidar.next()
