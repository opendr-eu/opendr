import numpy as np
from datetime import datetime

from os import PathLike
from pathlib import Path
from typing import Union, List

from opendr.engine.data import Image
from opendr.engine.datasets import ExternalDataset, DatasetIterator



class KittiDataset(ExternalDataset, DatasetIterator):

    def __init__(self, path: Union[str, Path, PathLike]):
        super().__init__(path, dataset_type='kitti')

        self._path = Path(path/ 'sequences')
    
        # ========================================================
        self.frame_ids = [0, -1, 1]
        self.scales = [0, 1, 2, 3]
        self.height = 192
        self.width = 640
        # self.sequences = ['00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10']
        self.sequences = ['00'] # for debugging
        # ========================================================

        self._images = {}
        self._velocities = {}
        self._timestamps = {}
        for sequence in self.sequences:
            self._image_folder = self._path / sequence /'image_2'
            self._velocity_folder = self._path / sequence /'oxts/data'
            self._timestamps_file = self._path / sequence /'oxst/timestamps.txt'

            self._images[sequence] = sorted(self._image_folder.glob(f'*.png'))
            self._velocities[sequence] = sorted(self._velocity_folder.glob(f'*.txt'))
            self._timestamps[sequence] = self._create_timestamps(timestamps=self._timestamps_file)

        self.camera_matrix = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)


   
    def _create_timestamps(self, timestamps) -> List[int]:
        """
        This method is used for creating a list of timestamps from the timestamps file.

        :return: a list of timestamps
        :rtype: List of float
        """
        timestamps = []
        with open(self._timestamps_file, 'r', encoding='utf-8') as f:
            string_timestamps = f.read().splitlines()
        # Convert relative to timestamp of initial frame and output in seconds
        # Discard nanoseconds as they are not supported by datetime
        for timestamp in string_timestamps:
            timestamps.append(
                (datetime.strptime(timestamp[:-3], '%Y-%m-%d %H:%M:%S.%f') -
                 datetime.strptime(string_timestamps[0][:-3], '%Y-%m-%d %H:%M:%S.%f')).total_seconds())
        return timestamps


    def __getitem__(self, idx):
        """
        This method is used for loading the idx-th sample of a dataset along with its annotation.
        In this case, the annotation is split up into different files and, thus, a different interface is used.

        :param idx: the index of the sample to load
        :type idx: int
        :return: the idx-th sample and a placeholder for the corresponding annotation
        :rtype: Tuple of (Image, None)
        
        """
        return super().__getitem__(idx)

