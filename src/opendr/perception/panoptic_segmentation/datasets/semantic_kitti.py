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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import sys
import shutil
import warnings

from tqdm import tqdm
import numpy as np

from mmdet2.datasets import CityscapesDataset as MmdetCityscapesDataset
from mmdet2.datasets import build_dataset
from mmdet2.datasets.eval_np import PanopticEval

from opendr.engine.data import PointCloud
from opendr.engine.datasets import ExternalDataset, DatasetIterator


# Palette values are taken from https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
PALETTE = np.array([
    [245, 150, 100],  # Car - 0
    [245, 230, 100],  # Bycicle - 1
    [250, 80, 100],  # Motorcycle - 2
    [180, 30, 80],   # Truck - 3
    [255, 0, 0],  # Other vehicle - 4
    [30, 30, 255],  # Person - 5
    [200, 40, 255],  # Bicyclist - 6
    [90, 30, 150],  # Motorcyclist - 7
    [255, 0, 255],  # Road - 8
    [255, 150, 255],  # Parking - 9
    [75, 0, 75],  # Sidewalk - 10
    [75, 0, 175],  # Other ground - 11
    [0, 200, 255],  # Building - 12
    [50, 120, 255],  # Fence - 13
    [0, 175, 0],  # Vegetation - 14
    [0, 60, 135],  # Trunk - 15
    [80, 240, 150],  # Terrain - 16
    [150, 240, 255],  # Pole - 17
    [0, 0, 255],  # Traffic sign - 18
    [0, 0, 0],  # Unlabelled - 18
])


class SemanticKittiDataset(ExternalDataset, DatasetIterator):
    """
    The SemanticKittiDataset class provides interfaces to the OpenDR and MMDetector API for different use cases.
    For inference using a pre-trained model, the OpenDR interface is supported.
    For training and evaluation, an instance of the MMDetector dataset is created.

    The Semantic KITTI dataset can be found on the official website: http://www.semantic-kitti.org/dataset.html
    The files:
            * KITTI Odometry Benchmark Velodyne point clouds
            * KITTI Odometry Benchmark Calibration data
            * SemanticKITTI label data
    must be downloaded and extracted into the same folder.

    The folder structure should look like:
    path
     └── sequences
          ├── 00
          │    ├── calib.txt
          │    ├── poses.txt
          │    ├── times.txt
          │    ├── labels
          │    │    ├── 000000.label
          │    │    ├──     ⋮
          │    │    └── 004540.label
          │    └── velodyne
          │         ├── 000000.bin
          │         ├──     ⋮
          │         └── 004540.bin
          ├── ⋮
          └── 21
               ├── calib.txt
           ⋮

    NOTE: Only sequences 00-10 have ground truth values and, thus are the only ones with the poses.txt file
    and labels/ directory.
    """

    def __init__(self,
                 path: Union[str, Path],
                 split: Optional[str]
                 ):
        """
        Constructor.

        :param path: Path to the KITTI dataset root. Parent of the sequences/ directory.
        :type path: str | Path
        :param split: Type of the data split. Valid values: ["train", "valid", "test"]. If None, then "train" will be used.
        :type split: str|None
        """

        super().__init__(path=str(path), dataset_type="SemanticKITTIDataset")

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset path {self.path} does not exist. \
                                     Please download the dataset according to the instructions in the README.md file.")

        self._pipeline = None
        self._mmdet2_dataset = (None, None)
        self._file_list = None
        self.split = split

    @property
    def pipeline(self
                 ) -> List[Dict[str, Any]]:
        """
        Getter of the data loading pipeline.

        :return: data loading pipeline.
        :rtype: list
        """
        return self._pipeline

    @pipeline.setter
    def pipeline(self,
                 value: List[Dict[str, Any]]
                 ):
        """
        Setter for the data loading pipeline.

        :param value: data loading pipeline.
        :type value: list
        """
        self._pipeline = value

    @property
    def split(self
              ) -> str:
        """
        Getter for the data split key.

        :return: Key of the dataset split.
        :rtype: str
        """
        return self._split

    @split.setter
    def split(self,
              value: str
              ):
        """
        Setter for the data split key.

        :param value: Type of the data split. Valid values: ["train", "valid", "test"]. If None, then "train" will be used.
        :type value: str|None
        """

        if value is None:
            self._split = "train"

        valid_values = ["train", "valid", "test"]

        value = value.lower()

        if value not in valid_values:
            raise ValueError(f"Invalid value for split. Valid values: {', '.join(valid_values)}")

        self._split = value

    def evaluate(self,
                 prediction_path: Union[Path, str],
                 min_points: int = 50,
                 ) -> Dict[str, Any]:
        """
        This method is used to evaluate the predictions versus the ground truth returns the following stats:
            - Panoptic Quality (PQ)
            - Segmentation Quality (SQ)
            - Recognition Quality (RQ)
            - Intersection over Union (IOU)

        This function contains modified code from '_evaluate_panoptic()' in
            src/opendr/perception/panoptic_segmentation/efficient_lps/algorithm/EfficientLPS/mmdet2/datasets/semantic_kitti.py

        :param prediction_path: path to the predicted stuffandthing maps.
        :type prediction_path: str | pathlib.Path
        :param min_points:
        :type min_points: int

        :return: Evaluation statistics.
        :rtype: dict
        """

        if not isinstance(prediction_path, Path):
            prediction_path = Path(prediction_path)

        if not prediction_path.exists():
            raise ValueError('The provided prediction_path does not exist.')

        dataset = self.get_mmdet2_dataset()
        ignore_class = [k for k, ignored in dataset.class_ignore.items() if ignored]
        # Redirect stdout to /dev/null for less verbosity
        prev_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        class_evaluator = PanopticEval(dataset.nr_classes, None, ignore_class, min_points=min_points)
        sys.stdout.close()
        sys.stdout = prev_stdout

        tst_sequences = dataset.cfg["split"][dataset.split]

        gt_files = sorted([f for s in tst_sequences
                           for f in (Path(dataset.ann_file) / "{0:02}".format(s) / "labels").rglob("*.label")])
        pred_files = sorted([f for s in tst_sequences
                             for f in (prediction_path / "{0:02}".format(s) / "predictions").rglob("*.label")])

        if len(gt_files) != len(pred_files):
            raise RuntimeError("Number of GT Label files does not match number of Predicted label files")

        for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
            gt_inst = np.fromfile(gt_file, dtype=np.uint32)
            gt_sem = dataset.class_lut[gt_inst & 0xFFFF]   # remap to xentropy format

            pred_inst = np.fromfile(pred_file, dtype=np.uint32)
            pred_sem = (pred_inst & 0xFFFF) + 1
            pred_sem[pred_sem == 256] = 0

            class_evaluator.addBatch(pred_sem, pred_inst, gt_sem, gt_inst)

        class_pq, class_sq, class_rq, class_all_pq, class_all_sq, class_all_rq = class_evaluator.getPQ()
        class_iou, class_all_iou = class_evaluator.getSemIoU()

        results_all = {"PQ": class_pq, "SQ": class_sq, "RQ": class_rq, "IoU": class_iou}
        results_cls = {"PQ": class_all_pq, "SQ": class_all_sq, "RQ": class_all_rq, "IoU": class_all_iou}

        things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
        stuff = ['road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence',
                 'pole', 'traffic-sign']
        all_idx = np.array(list(range(dataset.nr_classes)))
        class_dict = {dataset.class_strings[dataset.class_inv_remap[i]]: i for i in all_idx}
        things_idx = np.array([class_dict[k] for k in things], dtype=np.int)
        stuff_idx = np.array([class_dict[k] for k in stuff], dtype=np.int)

        results = {
            "all": results_all,
            "things": {k: np.mean(v[things_idx]).item() for k, v in results_cls.items()},
            "stuff": {k: np.mean(v[stuff_idx]).item() for k, v in results_cls.items()},
            "per_class": {k: {vk: v[i] for vk, v in results_cls.items()} for k, i in class_dict.items()}
        }

        return results

    @staticmethod
    def prepare_data(download_path: Union[Path, str],
                     dataset_path: Union[Path, str],
                     ) -> None:
        """
        This method is used to structure extracted zip of the dataset.

        :param download_path: Path to the downloaded and unzipped file.
        :type download_path: str | pathlib.Path
        :param dataset_path: Path to the directory where the dataset will be stored.
        :type dataset_path: str | pathlib.Path
        :param num_workers: Number of workers to use for parallel processing.
        :type num_workers: int
        """

        if not isinstance(download_path, Path):
            download_path = Path(download_path)

        if not isinstance(dataset_path, Path):
            dataset_path = Path(dataset_path)

        if not download_path.exists():
            raise ValueError('The provided download_path does not exist.')

        if not dataset_path.exists():
            dataset_path.mkdir(parents=True)

        # Create the structured dataset directory
        output_dir = dataset_path / 'semantic-kitti' / 'sequences'
        sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                     '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        subdir_names = ['data_odometry_calib', 'data_odometry_labels', 'data_odometry_velodyne']
        # Go sub-directories to extract the names
        sub_dirs = [f for f in download_path.iterdir() if f.is_dir()]
        sub_dirs.sort()
        # Check if the sub-directories are the ones we expect
        if not all([s.name in subdir_names for s in sub_dirs]):
            raise ValueError('The provided download_path does not contain the expected sub-directories.')
        # create 3 iterators for parallel processing. One for "Calib", one for "Labels" and one for "Velodyne"

        calib_iter = sub_dirs[0] / 'dataset' / 'sequences'
        label_iter = sub_dirs[1] / 'dataset' / 'sequences'
        velodyne_iter = sub_dirs[2] / 'dataset' / 'sequences'

        for sequence in tqdm(sequences):
            # Check if that sequence exists in the downloaded dataset
            if not (calib_iter / sequence).exists():
                warnings.warn('The sequence {} does not exist in the {}.'.format(sequence, subdir_names[0]))
                continue
            if not (label_iter / sequence).exists():
                warnings.warn('The sequence {} does not exist in the {}.'.format(sequence, subdir_names[1]))
                continue
            if not (velodyne_iter / sequence).exists():
                warnings.warn('The sequence {} does not exist in the {}.'.format(sequence, subdir_names[2]))
                continue
            # Create the sequence directory
            seq_dir = output_dir / sequence
            seq_dir.mkdir(parents=True)

            # Copy the files
            for file in (calib_iter / sequence).iterdir():
                shutil.move(str(file), str(seq_dir))
            for file in (label_iter / sequence).iterdir():
                shutil.move(str(file), str(seq_dir))
            for file in (velodyne_iter / sequence).iterdir():
                shutil.move(str(file), str(seq_dir))

    def __getitem__(self,
                    idx: int
                    ) -> Tuple[PointCloud, None]:
        """
        Method is used for loading the idx-th sample of a dataset along with its annotation.
        In this case, the annotation is split up into different files and, thus, a different interface is used.

        :param idx: Index of the sample to load.
        :type idx: int

        :return: The idx-th sample and the corresponding annotation.
        :rtype: Tuple of (PointCloud, None)
        """

        dataset = self._get_mmdet2_dataset(test_mode=not self.split == "train", ignore_pipeline=True)
        item_path = dataset.vel_seq_infos[idx]
        point_cloud = self.load_point_cloud(item_path)

        return point_cloud, None

    def __len__(self
                ) -> int:
        """
        This method returns the size of the dataset in terms of number of data points (scan frames).

        :return: the size of the dataset
        :rtype: int
        """

        return len(self._get_mmdet2_dataset(test_mode=not self.split == "train", ignore_pipeline=True))

    def get_mmdet2_dataset(self,
                           test_mode: bool = False,
                           ) -> MmdetCityscapesDataset:
        """
        Returns the dataset in a format compatible with the mmdet2 dataloader.

        :param test_mode: Whether to use the train or test data pipelines.
                          If set to True, the panoptic ground truth data has to be present
        :type test_mode: bool

        :return: MMDet compatible dataset
        :rtype: MmdetCityscapesDataset
        """

        return self._get_mmdet2_dataset(test_mode=test_mode, ignore_pipeline=False)

    def _get_mmdet2_dataset(self,
                            test_mode: bool = False,
                            ignore_pipeline: bool = False,
                            ) -> MmdetCityscapesDataset:
        """
        Private version of the get_mmdet2_dataset that can ignore the pipeline.

        :param test_mode: Whether to use the train or test data pipelines.
                                          If set to True, the panoptic ground truth data has to be present
        :type test_mode: bool
        :param ignore_pipeline: Ignores the fact that no pipeline has been set. Used by the iterator methods.
                                Use caution when setting it to True.
        :type ignore_pipeline: bool

        :return: MMDet compatible dataset
        :rtype: MmdetCityscapesDataset
        """

        if self._mmdet2_dataset[0] is None or self._mmdet2_dataset[1] != test_mode:
            self._mmdet2_dataset = (self._build_mmdet2_dataset(test_mode, ignore_pipeline=ignore_pipeline), test_mode)
        return self._mmdet2_dataset[0]

    def _build_mmdet2_dataset(self,
                              test_mode: bool = False,
                              ignore_pipeline: bool = False,
                              ) -> MmdetCityscapesDataset:
        """
        Generates the mmdet2 representation of the dataset to be used with the mmdet2 API.

        :param test_mode: Whether to use the train or test data pipelines.
                                          If set to True, the panoptic ground truth data has to be present
        :type test_mode: bool
        :param ignore_pipeline: Ignores the fact that no pipeline has been set. Used by the iterator methods.
                                Use caution when setting it to True.
        :type ignore_pipeline: bool

        :return: MMDet compatible dataset
        :rtype: MmdetCityscapesDataset
        """

        if not self.pipeline:
            if ignore_pipeline:
                self.pipeline = []
            else:
                raise ValueError("No dataset pipeline has been set.")

        config_path = Path(__file__).parent.parent / "efficient_lps" / "algorithm" / "EfficientLPS" / "configs"

        cfg = {
            "type": self.dataset_type,
            "ann_file": Path(self.path) / "sequences",
            "config": config_path / "semantic-kitti.yaml",
            "split": self.split,
            "pipeline": self.pipeline
        }

        if test_mode:
            mmdet2_dataset = build_dataset(cfg, {"test_mode": test_mode})
        else:
            mmdet2_dataset = build_dataset(cfg)

        return mmdet2_dataset

    @staticmethod
    def load_point_cloud(path):
        return PointCloud(data=np.fromfile(path, dtype=np.float32).reshape(-1, 4))
