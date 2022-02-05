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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys

import numpy as np
from tqdm import tqdm

from mmdet.datasets import CityscapesDataset as MmdetCityscapesDataset
from mmdet.datasets import build_dataset
from mmdet.datasets.eval_np import PanopticEval

from opendr.engine.data import PointCloud
from opendr.engine.datasets import ExternalDataset, DatasetIterator


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

	NOTE: Only sequences 00-10 have ground truth values and, thus are the only ones with the poses.txt and labels/.
	"""

	def __init__(self, path: str,
				 split: Optional[str]):
		"""
		Constructor.

		:param path: Path to the KITTI dataset root. Parent of the sequences/ directory.
		:type path: str
		:param split: Type of the data split. Valid values: ["train", "valid", "test"]. If None, then "train" will be used.
		:type split: str|None
		"""

		super().__init__(path=path, dataset_type="SemanticKITTIDataset")

		self._pipeline = None
		self._mmdet_dataset = (None, None)
		self.split = split

	@property
	def pipeline(self) -> List[Dict[str, Any]]:
		"""
		Getter of the data loading pipeline.

		:return: data loading pipeline
		:rtype: list
		"""
		return self._pipeline

	@pipeline.setter
	def pipeline(self, value: List[Dict[str, Any]]):
		"""
		Setter for the data loading pipeline

		:param value: data loading pipeline
		:type value: list
		"""
		self._pipeline = value

	@property
	def split(self) -> str:
		"""
		TODO:

		:return:
		"""
		return self._split

	@split.setter
	def split(self, value: str):

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
		TODO:

		"""
		if isinstance(prediction_path, str):
			prediction_path = Path(prediction_path)

		if not prediction_path.exists():
			raise ValueError('The provided prediction_path does not exist.')

		dataset = self.get_mmdet_dataset()
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
		resuls_cls = {"PQ": class_all_pq, "SQ": class_all_sq, "RQ": class_all_rq, "IoU": class_all_iou}

		things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
		stuff = [ 'road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence',
				  'pole', 'traffic-sign']
		all_idx = np.array(list(range(dataset.nr_classes)))
		class_dict = {dataset.class_strings[dataset.class_inv_remap[i]]: i for i in all_idx}
		things_idx = np.array([class_dict[k] for k in things], dtype=np.int)
		stuff_idx = np.array([class_dict[k] for k in stuff], dtype=np.int)

		results = {
			"all": results_all,
			"things": { k: np.mean(v[things_idx]).item() for k, v in resuls_cls.items()},
			"stuff": { k: np.mean(v[stuff_idx]).item() for k, v in resuls_cls.items()},
			"per_class": { k: {vk: v[i] for vk, v in resuls_cls.items()} for k, i in class_dict.items()}
		}

		return results

	def __getitem__(self, idx):
		"""
		TODO
		"""
		raise NotImplementedError

	def __len__(self):
		"""
		TODO
		:return:
		"""
		raise NotImplementedError

	def get_mmdet_dataset(self,
						  test_mode: bool = False
						  ) -> MmdetCityscapesDataset:
		"""
		TODO
		:param test_mode:
		:type test_mode: bool

		:return:
		"""
		if self._mmdet_dataset[0] is None or self._mmdet_dataset[1] != test_mode:
			self._mmdet_dataset = (self._build_mmdet_dataset(test_mode), test_mode)
		return self._mmdet_dataset[0]

	def _build_mmdet_dataset(self,
							 test_mode: bool = False
							 ) -> MmdetCityscapesDataset:
		"""
		TODO

		:param test_mode:
		:type test_mode: bool
		:return:
		"""
		if self.pipeline is None:
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
			mmdet_dataset = build_dataset(cfg, {"test_mode": test_mode})
		else:
			mmdet_dataset = build_dataset(cfg)

		return mmdet_dataset
