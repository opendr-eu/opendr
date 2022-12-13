# Copyright 2020-2022 OpenDR European Project
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

import bisect
from itertools import accumulate

from opendr.engine.datasets import DatasetIterator


class DetectionDataset(DatasetIterator):
    def __init__(self, classes, dataset_type, root, image_paths=None, splits='',
                 image_transform=None, target_transform=None, transform=None):
        super().__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.image_paths = image_paths
        self.dataset_type = dataset_type
        self.root = root
        self.splits = splits

        self._transform = transform
        self._image_transform = image_transform
        self._target_transform = target_transform

    def set_transform(self, transform):
        self._transform = transform

    def transform(self, transform):
        return MappedDetectionDataset(self, transform)

    def set_image_transform(self, transform):
        self._image_transform = transform

    def set_target_transform(self, transform):
        self._target_transform = transform

    def get_bboxes(self, item):
        pass

    def get_image(self, item):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MappedDetectionDataset(DatasetIterator):
    def __init__(self, data, map_function):
        self.data = data
        self.map_function = map_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item, tuple):
            return self.map_function(*item)
        return self.map_function(item)


class ConcatDataset(DetectionDataset):
    """
    Basic dataset concatenation class. The datasets are assumed to have the same classes.

    :param datasets: list of DetectionDataset type or subclass
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(classes=datasets[0].classes, dataset_type='concat_dataset',
                                            root=None)
        self.cumulative_lengths = list(accumulate([len(dataset) for dataset in datasets]))
        self.datasets = datasets

    def set_transform(self, transform):
        self._transform = transform
        for dataset in self.datasets:
            dataset.transform(transform)

    def transform(self, transform):
        mapped_datasets = [MappedDetectionDataset(dataset, transform) for dataset in self.datasets]
        return ConcatDataset(mapped_datasets)

    def set_image_transform(self, transform):
        self._image_transform = transform
        for dataset in self.datasets:
            dataset.set_image_transform(transform)

    def set_target_transform(self, transform):
        self._target_transform = transform
        for dataset in self.datasets:
            dataset.set_target_transform(transform)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, item):
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, item)
        if dataset_idx == 0:
            sample_idx = item
        else:
            sample_idx = item - self.cumulative_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


def is_image_type(filename):
    return filename.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'))


def remove_extension(filename):
    return '.'.join(filename.split('.')[:-1])
