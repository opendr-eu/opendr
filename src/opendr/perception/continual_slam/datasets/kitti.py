import numpy as np
import os
import re
from datetime import datetime

from os import PathLike
from pathlib import Path
from typing import Tuple, Union, Any, List

from opendr.engine.data import Image
from opendr.engine.datasets import ExternalDataset, DatasetIterator



class KittiDataset(ExternalDataset, DatasetIterator):

    def __init__(self, path: Union[str, Path, PathLike]):
        super().__init__(path, dataset_type='kitti')

        self._path = Path(os.path.join(path, 'sequence'))
    
        # ========================================================
        self.frame_ids = [0, -1, 1]
        self.scales = [0, 1, 2, 3]
        self.height = 192
        self.width = 640
        # self.sequences = ['00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10']
        self.sequences = os.listdir(self._path)

        if self.sequences is None:
            raise FileNotFoundError(f'No sequences found in {self._path}')
        # ========================================================

        self._images = {}
        self._velocities = {}
        self._timestamps = {}
        for sequence in self.sequences:
            self._image_folder = self._path / sequence /'image_2'
            self._velocity_folder = self._path / sequence /'oxts/data'
            self._timestamps_file = self._path / sequence /'oxts/timestamps.txt'

            self._images[sequence] = sorted(self._image_folder.glob(f'*.png'))
            self._velocities[sequence] = sorted(self._velocity_folder.glob(f'*.txt'))
            self._timestamps[sequence] = self._create_timestamps(timestamps=self._timestamps_file)

        self.camera_matrix = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        # Now we simply put all sequences available on single lists
        self._create_lists_from_sequences()

   
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


    def _load_relative_distance(self, index: int) -> float:
        """
        Distance in meters and with respect to the previous frame.
        """
        previous_timestamp = self.timestamps[index - 1]
        current_timestamp = self.timestamps[index]
        delta_timestamp = current_timestamp - previous_timestamp  # s
        # Compute speed as norm of forward, leftward, and upward elements
        previous_speed = np.linalg.norm(np.loadtxt(str(self.velocities[index - 1]))[8:11])
        current_speed = np.linalg.norm(np.loadtxt(str(self.velocities[index]))[8:11])
        speed = (previous_speed + current_speed) / 2  # m/s
        distance = speed * delta_timestamp
        return distance, speed

    def _create_lists_from_sequences(self):
        """
        This method is used for creating a single list of images, velocities and timestamps from the sequences.
        """

        self.images = []
        self.velocities = []
        self.timestamps = []

        for sequence in self.sequences:
            self.images.extend(self._images[sequence])
            self.velocities.extend(self._velocities[sequence])
            self.timestamps.extend(self._timestamps[sequence])

    def __getitem__(self, idx: int) -> Tuple[Any, None]:
        """
        This method is used for getting the item at the given index.

        :param idx: index of the item
        :type idx: int
        :return: the item at the given index
        :rtype: Tuple[Any, None]
        """
        data = {}
        for i in range(idx-2, idx+1):
            image = Image.open(str(self.images[i]))
            distance, speed = self._load_relative_distance(i)
            image_id = self.images[i].name.split('.')[0]
            sequence_id = re.findall("sequence/\d\d", str(self.images[i]))[0].split('/')[1]
            image_id = sequence_id + '_' + image_id
            data[image_id] = (image, distance, speed)
        return data, None

    def __len__(self):
        """
        This method is used for getting the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.images)

if __name__ == '__main__':
    kitti = KittiDataset(path='/home/canakcia/Desktop/')
    for i in range(len(kitti)):
        x = kitti[i]
        

