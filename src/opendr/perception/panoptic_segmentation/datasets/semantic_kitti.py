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

from pathlib import Path
from typing import Any, Dict, List, Union

from mmdet.datasets import CityscapesDataset as MmdetCityscapesDataset
from mmdet.datasets import build_dataset

from opendr.engine.data import PointCloud
from opendr.engine.datasets import ExternalDataset, DatasetIterator


class SemanticKittiDataset(ExternalDataset, DatasetIterator):
	"""
	TODO
	:param split: Type of the data split. Valid values: "train", "valid", "test"
	:type split: str|None
	"""
	def __init__(self, path: str,
				 split: Union[str, None]):

		super().__init__(path=path, dataset_type="SemanticKITTIDataset")

		self._pipeline = None
		self._mmdet_dataset = (None, None)
		self.split = split

	@property
	def pipeline(self) -> List[dict]:
		"""
		Getter of the data loading pipeline.

		:return: data loading pipeline
		:rtype: list
		"""
		return self._pipeline

	@pipeline.setter
	def pipeline(self, value):
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
				 prediction_json_folder: Union[Path, str]
				 ) -> Dict[str, Any]:
		"""
		TODO:

		"""
		raise NotImplementedError

	def __getitem__(self, idx):
		raise NotImplementedError

	def __len__(self):
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
