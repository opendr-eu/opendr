from time import sleep
import cv2
import numpy as np
from opendr.engine.data import Image
from opendr.perception.object_tracking_2d.datasets.mot_dataset import RawMotDatasetIterator, RawMotWithDetectionsDatasetIterator


def disk_image_generator(path, splits, count=None, cycle=True):
    dataset = RawMotDatasetIterator(path, splits, scan_labels=False)

    i = 0

    len_dataset = len(dataset) if count is None else count

    while i < len_dataset or cycle:
        yield dataset[i % len_dataset][0]
        i += 1


def disk_image_with_detections_generator(path, splits, count=None, cycle=True):
    dataset = RawMotWithDetectionsDatasetIterator(path, splits, (1920, 1080))

    i = 0

    len_dataset = len(dataset) if count is None else count

    while i < len_dataset or cycle:
        yield dataset[i % len_dataset][0]
        i += 1


def camera_image_generator(video_source):

    while True:
        image = video_source.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(np.moveaxis(image, [0, 1, 2], [1, 2, 0]))
        # (480, 640, 3)
        # (3, 1080, 1920)
        yield Image(image)
