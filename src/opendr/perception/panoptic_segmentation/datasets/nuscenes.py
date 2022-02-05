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

from opendr.engine.datasets import ExternalDataset, DatasetIterator


class NuscenesDataset(ExternalDataset):
	def __init__(self, path: str):

		super().__init__(path=path, dataset_type="nuscenes")

	def evaluate(self
				 ):
		raise NotImplemented

	def __getitem__(self, idx):
		raise NotImplemented

	def __len__(self):
		raise NotImplemented

	def get_mmdet_dataset(self, test_mode=False):
		raise NotImplemented
