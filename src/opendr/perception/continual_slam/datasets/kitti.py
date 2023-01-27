import numpy as np
import os
import re
from datetime import datetime

from os import PathLike
from pathlib import Path
from typing import Tuple, Union, Any, List

from opendr.engine.data import Image
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from opendr.perception.continual_slam.datasets.config import Dataset

from torchvision.transforms import Resize, InterpolationMode
from PIL import Image as PILImage


class KittiDataset(ExternalDataset, DatasetIterator):

    def __init__(self, path: Union[str, Path, PathLike], config: Dataset):
        super().__init__(path, dataset_type='kitti')

        self._path = Path(os.path.join(path, 'sequences'))

        # ========================================================
        self.frame_ids = config.frame_ids
        self.height = config.height
        self.width = config.width
        self.scales = config.scales
        self.valid_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        # self.valid_sequences = ['09']
        self.sequences = os.listdir(self._path)

        if self.sequences is None:
            raise FileNotFoundError(f'No sequences found in {self._path}')
        # ========================================================

        self._images = {}
        self._velocities = {}
        self._timestamps = {}
        for sequence in self.sequences:
            # Check if the sequence is valid
            if sequence not in self.valid_sequences:
                print(f'Sequence {sequence} is not valid. Skipping...')
                continue
            self._image_folder = self._path / sequence / 'image_2'
            self._velocity_folder = self._path / sequence / 'oxts/data'
            self._timestamps_file = self._path / sequence / 'oxts/timestamps.txt'

            self._images[sequence] = sorted(self._image_folder.glob(f'*.png'))
            self._velocities[sequence] = sorted(self._velocity_folder.glob(f'*.txt'))
            self._timestamps[sequence] = self._create_timestamps(timestamps=self._timestamps_file)

        self.camera_matrix = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        # Now we simply put all sequences available on into a single list
        self._create_lists_from_sequences()

        self.sequence_check = None

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
        return distance

    def _create_lists_from_sequences(self):
        """
        This method is used for creating a single list of images, velocities and timestamps from the sequences.
        """

        self.images = []
        self.velocities = []
        self.timestamps = []

        for sequence in self.sequences:
            # Check if the sequence is valid
            if sequence not in self.valid_sequences:
                # print(f'Sequence {sequence} is not valid. Skipping...')
                continue
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
        flag = False
        counter = 0
        for i in range(idx, idx+3):
            image = PILImage.open(str(self.images[i]))
            image = Resize((self.height, self.width), interpolation=InterpolationMode.LANCZOS)(image)
            image = Image(image).opencv().transpose(2, 0, 1)
            image = Image(image)
            distance = self._load_relative_distance(i)
            image_id = self.images[i].name.split('.')[0]
            sequence_id = re.findall("sequences/\d\d", str(self.images[i]))[0].split('/')[1]
            if self.sequence_check == None:
                self.sequence_check = sequence_id
            else:
                if self.sequence_check != sequence_id:
                    # This means that now we have a new sequence and we need to wait two iterations
                    # to get the first image of the new sequence
                    flag = True
            if flag:
                counter += 1
                if counter < 2:
                    continue
                else:
                    self.sequence_check = sequence_id
                    flag = False
                    counter = 0
            image_id = sequence_id + '_' + image_id
            data[image_id] = (image, distance)
        return data, None

    def __len__(self):
        """
        This method is used for getting the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.images)

# TODO: For debugging purposes only
if __name__ == '__main__':
    local_path = Path(__file__).parent.parent / 'configs'
    from opendr.perception.continual_slam.configs.config_parser import ConfigParser
    dataset_config = ConfigParser(local_path / 'singlegpu_kitti.yaml').dataset
    dataset_path = dataset_config.dataset_path
    dataset = KittiDataset(str(dataset_path), dataset_config)
    for i in range(len(dataset)):
        print(dataset[i][0].keys())