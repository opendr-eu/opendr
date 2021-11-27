# Copyright 2020 Aristotle University of Thessaloniki
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

from abc import ABC, abstractmethod
import os
import numpy as np
from opendr.engine.data import PointCloud


class Dataset(ABC):
    """
    Dataset abstract class is used to provide a common ancestor type for dataset classes.
    All classes that provide dataset-related functionality must inherit this class.
    """

    def __init__(self):
        pass


class DatasetIterator(Dataset):
    """
    DatasetIterator serves as an abstraction layer over the different types of datasets.
    In this way it provides the opportunity to users to implement different kinds of datasets, while
    providing a uniform interface.

    This class provides the following abstract methods:
    - __getitem__(i), a getter that allows for retrieving the i-th sample of the dataset, along with its annotation
    - __len__(), which allows for getting the size of the dataset
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx):
        """
        This method is used for loading the idx-th sample of a dataset along with its annotation.

        :param idx: the index of the sample to load
        :return: the idx-th sample and its annotation
        :rtype: Tuple of (Data, Target)
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        This method returns the size of the dataset.

        :return: the size of the dataset
        :rtype: int
        """
        pass


class MappedDatasetIterator(DatasetIterator):
    """
    MappedDatasetIterator allows to transform elements of the original DatasetIterator.

    This class provides the following methods:
    - __getitem__(i), a getter that allows for retrieving the i-th sample of the dataset, along with its annotation
    - __len__(), which allows for getting the size of the dataset
    """
    def __init__(self, original, map_function):
        super().__init__()
        self.map_function = map_function
        self.original = original

    def __getitem__(self, idx):
        """
        This method is used for loading the idx-th sample of a dataset along with its annotation.

        :param idx: the index of the sample to load
        :return: the idx-th sample and its annotation
        :rtype: Tuple of (Data, Target)
        """

        return self.map_function(self.original[idx])

    def __len__(self):
        """
        This method returns the size of the dataset.

        :return: the size of the dataset
        :rtype: int
        """

        return len(self.original)


class ExternalDataset(Dataset):
    """
    ExternalDataset provides a way for handling well-known external dataset formats
    (e.g., COCO, PascalVOC, Imagenet, etc.) directly by OpenDR, without requiring any special effort
    from the users for writing a specific loader.

    Class attributes:
    - path: path to the external dataset
    - dataset_type:	type of the dataset format (e.g., “voc”, “coco”, “imagenet”, etc.)
    """
    def __init__(self, path, dataset_type):
        super().__init__()
        self.path = path
        self.dataset_type = dataset_type

    @property
    def path(self):
        """
        Getter of path field.
        This returns the path.

        :return: path to the external dataset
        :rtype: str
        """
        return self._path

    @path.setter
    def path(self, value):
        """
        Setter of path field. This will perform the necessary type checking.

        :param value: new path value
        :type value: str
        """
        if type(value) != str:
            raise ValueError('path should be a str')
        else:
            self._path = value

    @property
    def dataset_type(self):
        """
        Getter of dataset_type field.
        This returns the dataset_type.

        :return: type of the dataset format (e.g., “voc”, “coco”, “imagenet”, etc.)
        :rtype: str
        """
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, value):
        """
        Setter of dataset_type field. This will perform the necessary type checking.

        :param value: new dataset_type value
        :type value: str
        """
        if type(value) != str:
            raise ValueError('dataset_type should be a str')
        else:
            self._dataset_type = value


class PointCloudsDatasetIterator(DatasetIterator):
    def __init__(self, path, num_point_features=4):
        super().__init__()

        self.path = path
        self.num_point_features = num_point_features
        self.files = sorted(os.listdir(path))

    def __getitem__(self, idx):
        data = np.fromfile(
            str(self.path + "/" + self.files[idx]), dtype=np.float32, count=-1
        ).reshape([-1, self.num_point_features])

        return PointCloud(data)

    def __len__(self):
        return len(self.files)
