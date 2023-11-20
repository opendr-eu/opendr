from typing import List
import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    A learnable scale parameter
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class MultiOutput(nn.Module):
    # Output a list of tensors
    def __init__(self):
        super(MultiOutput, self).__init__()

    def forward(self, x):
        outs = [out for out in x]
        return outs


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, self.d)


class Flatten(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.s = start_dim
        self.e = end_dim

    def forward(self, x):
        return torch.flatten(x, start_dim=self.s, end_dim=self.e)
