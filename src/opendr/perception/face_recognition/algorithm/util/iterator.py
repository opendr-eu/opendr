from opendr.engine.datasets import DatasetIterator
import torch
import numpy as np


class FaceRecognitionDataset(DatasetIterator):
    def __init__(self, pairs):
        self.data = np.array([pairs[:, 0], pairs[:, 1]])
        self.labels = pairs[:, 2]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image1 = self.data[0][idx]
        image2 = self.data[1][idx]
        label = self.labels[idx]
        sample = {'image1': image1, 'image2': image2, 'label': label}
        return sample

    def __len__(self):
        return len(self.labels)
