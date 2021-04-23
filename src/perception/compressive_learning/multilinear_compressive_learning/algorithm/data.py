import torch
import numpy as np


class DataWrapper:
    def __init__(self, opendr_dataset):
        self.dataset = opendr_dataset

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset.__getitem__(i)
        # change from rows x cols x channels to channels x rows x cols
        x = np.transpose(x.numpy(), axes=(2, 0, 1))
        return torch.from_numpy(x).float(), torch.tensor([y.data, ]).long()
