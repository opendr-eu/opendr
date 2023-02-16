# Copyright 2020-2023 OpenDR European Project
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

import numpy as np
import os
import re
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile

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

        if not path.exists():
            print(f'Given path {path} does not exist. Using path from config file...')
            path = config.dataset_path
        self._path = Path(os.path.join(path, 'sequences'))

        # ========================================================
        self.frame_ids = config.frame_ids
        self.height = config.height
        self.width = config.width
        self.scales = config.scales
        self.valid_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

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

    @staticmethod
    def prepare_dataset(raw_dataset_path: str = None,
                        odometry_dataset_path: str = None,
                        oxts: bool = True,
                        gt_depth: bool = False) -> None:
        """
        This method is used for preparing the dataset that is downloaded from the KITTI website.
        :param raw_dataset_path: path to the downloaded dataset
        :type raw_dataset_path: str
        :param odometry_dataset_path: path to the extracted dataset
        :type odometry_dataset_path: str
        :param oxts: whether to copy the oxts files
        :type oxts: bool
        :param gt_depth: whether to copy the ground truth depth files
        :type gt_depth: bool
        """

        # yapf: disable
        # Mapping between KITTI Raw Drives and KITTI Odometry Sequences
        KITTI_RAW_SEQ_MAPPING = {
            0: {'date': '2011_10_03', 'drive': 27, 'start_frame': 0, 'end_frame': 4540},
            1: {'date': '2011_10_03', 'drive': 42, 'start_frame': 0, 'end_frame': 1100},
            2: {'date': '2011_10_03', 'drive': 34, 'start_frame': 0, 'end_frame': 4660},
            # 3:  {'date': '2011_09_26', 'drive': 67, 'start_frame':    0, 'end_frame':  800}, # No IMU
            4: {'date': '2011_09_30', 'drive': 16, 'start_frame': 0, 'end_frame': 270},
            5: {'date': '2011_09_30', 'drive': 18, 'start_frame': 0, 'end_frame': 2760},
            6: {'date': '2011_09_30', 'drive': 20, 'start_frame': 0, 'end_frame': 1100},
            7: {'date': '2011_09_30', 'drive': 27, 'start_frame': 0, 'end_frame': 1100},
            8: {'date': '2011_09_30', 'drive': 28, 'start_frame': 1100, 'end_frame': 5170},
            9: {'date': '2011_09_30', 'drive': 33, 'start_frame': 0, 'end_frame': 1590},
            10: {'date': '2011_09_30', 'drive': 34, 'start_frame': 0, 'end_frame': 1200},
        }
        # yapf: enable

        total_frames = 0
        for mapping in KITTI_RAW_SEQ_MAPPING.values():
            total_frames += (mapping['end_frame'] - mapping['start_frame'] + 1)

        if gt_depth:
            with tqdm(desc='Copying depth files', total=total_frames * 2, unit='files') as pbar:
                for sequence, mapping in KITTI_RAW_SEQ_MAPPING.items():
                    # This is the improved gt depth using 5 consecutive frames from
                    # "Sparsity invariant CNNs", J. Uhrig et al., 3DV, 2017.
                    odometry_sequence_path = odometry_dataset_path / f'{sequence:02}' / 'gt_depth'
                    split = 'val' if sequence == 4 else 'train'
                    raw_sequence_path = raw_dataset_path / split / \
                                        f'{mapping["date"]}_drive_{mapping["drive"]:04}_sync' / \
                                        'proj_depth' / 'groundtruth'
                    if not raw_sequence_path.exists():
                        continue
                    for image in ['image_02', 'image_03']:
                        image_raw_sequence_path = raw_sequence_path / image
                        (odometry_sequence_path / image).mkdir(exist_ok=True, parents=True)
                        raw_filenames = sorted(image_raw_sequence_path.glob('*'))
                        for raw_filename in raw_filenames:
                            odometry_filename = odometry_sequence_path / image / raw_filename.name
                            frame = int(raw_filename.stem)
                            if mapping['start_frame'] <= frame <= mapping['end_frame']:
                                copyfile(raw_filename, odometry_filename)
                                pbar.update(1)
                                pbar.set_postfix({'sequence': sequence})

        if oxts:
            with tqdm(desc='Copying OXTS files', total=total_frames, unit='files') as pbar:
                for sequence, mapping in KITTI_RAW_SEQ_MAPPING.items():
                    odometry_sequence_path = odometry_dataset_path / f'{sequence:02}' / 'oxts'
                    raw_sequence_path = raw_dataset_path / \
                                        f'{mapping["date"]}' / \
                                        f'{mapping["date"]}_drive_{mapping["drive"]:04}_sync' / \
                                        'oxts'
                    if not raw_sequence_path.exists():
                        continue
                    odometry_sequence_path.mkdir(exist_ok=True, parents=True)
                    copyfile(raw_sequence_path / 'dataformat.txt',
                            odometry_sequence_path / 'dataformat.txt')
                    with open(raw_sequence_path / 'timestamps.txt', 'r', encoding='utf-8') as f:
                        timestamps = f.readlines()[mapping['start_frame']:mapping['end_frame'] + 1]
                    with open(odometry_sequence_path / 'timestamps.txt', 'w', encoding='utf-8') as f:
                        f.writelines(timestamps)

                    for image in ['data']:
                        image_raw_sequence_path = raw_sequence_path / image
                        (odometry_sequence_path / image).mkdir(exist_ok=True, parents=True)
                        raw_filenames = sorted(image_raw_sequence_path.glob('*'))
                        for raw_filename in raw_filenames:
                            odometry_filename = odometry_sequence_path / image / raw_filename.name
                            frame = int(raw_filename.stem)
                            if mapping['start_frame'] <= frame <= mapping['end_frame']:
                                copyfile(raw_filename, odometry_filename)
                                pbar.update(1)
                                pbar.set_postfix({'sequence': sequence})

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