# Copyright 2020-2021 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import random
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

# Imports for DatasetFromSampler and DistributedSamplerWrapper
from typing import Optional
from operator import itemgetter


class RandomSampler(Sampler):
    def __init__(self, dataset, seed=None):
        self.dataset = dataset
        if seed is not None:
            self.seed = seed

    def set_seed(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            random.seed(time.process_time())
            self.seed = random.randint(0, 2**32 - 1)
        return self.seed

    def __iter__(self):
        n = len(self.dataset)
        indexes = list(range(n))
        random.Random(self.seed).shuffle(indexes)
        return iter(indexes)

    def __len__(self):
        return len(self.dataset)


class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    # dataset: Sized

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
