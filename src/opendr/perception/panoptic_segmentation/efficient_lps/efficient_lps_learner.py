# Copyright 2020-2021 OpenDR European Project
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

import logging
from pathlib import Path
import shutil
import time
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings

from mmcv import Config
from mmcv.parallel import scatter, collate, MMDataParallel
from mmcv.runner import load_checkpoint, save_checkpoint, Runner, TextLoggerHook

from opendr.engine.learners import Learner
from opendr.engine.data import PointCloud

from opendr.perception.panoptic_segmentation.datasets import SemanticKittiDataset, NuscenesDataset

from .algorithm.EfficientLPS.mmdet.apis import single_gpu_test
from .algorithm.EfficientLPS.mmdet.apis.train import batch_processor
from .algorithm.EfficientLPS.mmdet.core import get_classes, build_optimizer, EvalHook
from .algorithm.EfficientLPS.mmdet.datasets import build_dataloader
from .algorithm.EfficientLPS.mmdet.datasets.pipelines import Compose
from .algorithm.EfficientLPS.mmdet.models import build_detector
from .algorithm.EfficientLPS.mmdet.utils import collect_env, get_root_logger


class EfficientLpsLearner(Learner):
	"""
	The EfficientLpsLearner class provides the top-level API to training and evaluating the EfficientLPS network.
	It particularly facilitates easy inference on Point Cloud inputs when using pre-trained model weights.
	"""

	def __init__(self,
				 lr: float = 0.07,  # TODO: Verify
				 iters: int = 160,  # TODO: Verify
				 batch_size: int = 1,  # TODO: Verify
				 optimizer: str = "SGD",
				 lr_schedule: Optional[Dict[str, Any]] = None,
				 momentum: float = 0.9,  # TODO: Verify
				 weight_decay: float = 0.0001,  # TODO: Verify
				 optimizer_config: Optional[Dict[str, Any]] = None,
				 checkpoint_after_iter: int = 1,  # TODO: Verify
				 temp_path: str = str(Path(__file__).parent / "eval_tmp_dir"),
				 device: str = "cuda:0",
				 num_workers: int = 1,  # TODO: Verify
				 seed: Optional[float] = None,
				 config_file: str = str(Path(__file__).parent / "configs" / "singlegpu_sample.py")
				 ):
		"""
		TODO DOC

		:param lr:
		:type lr:
		:param iters:
		:param batch_size:
		:param optimizer:
		:param lr_schedule:
		:param momentum:
		:param weight_decay:
		:param optimizer_config:
		:param checkpoint_after_iter:
		:param temp_path:
		:param device:
		:param num_workers:
		:param seed:
		:param config_file:
		"""

		super().__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer, temp_path=temp_path,
						 device=device)

		if lr_schedule is None:
			lr_schedule = {
				'policy': 'step',
				'warmup': 'linear',
				'warmup_iters': 500,
				'warmup_ratio': 1 / 3,
				'step': [120, 144]
			}

		self._lr_schedule = lr_schedule

		if optimizer_config is None:
			optimizer_config = {
				'grad_clip': {
					'max_norm': 35,
					'norm_type': 2
				}
			}

		self._checkpoint_after_iter = checkpoint_after_iter
		self._num_workers = num_workers

		self._cfg = Config.fromfile(config_file)
		self._cfg.workflow = [('train', 1)]
		self._cfg.model.pretrained = None
		self._cfg.optimizer = {
			'type': self.optimizer,
			'lr': self.lr,
			'momentum': momentum,
			'weight_decay': weight_decay
		}
		self._cfg.optimizer_config = optimizer_config
		self._cfg.lr_config = self.lr_schedule
		self._cfg.total_epochs = self.iters
		self._cfg.checkpoint_config = {'interval': self.checkpoint_after_iter}
		self._cfg.log_config = {
			'interval': 1,
			'hooks': [
				{'type': 'TextLoggerHook'},
				{'type': 'TensorboardLoggerHook'}
			]
		}
		self._cfg.gpus = 1
		self._cfg.seed = seed

		# Create Model
		self.model = build_detector(self._cfg.model, train_cfg=self._cfg.train_cfg, test_cfg=self._cfg.test_cfg)
		self.model.to(self.device)
		self._is_model_trained = False

	def __del__(self):
		"""
		TODO DOC
		:return:
		"""
		shutil.rmtree(self.temp_path, ignore_errors=True)

	def fit(self,
			dataset: Union[SemanticKittiDataset, NuscenesDataset],
			val_dataset: Optional[Union[SemanticKittiDataset, NuscenesDataset]] = None,
			logging_path: str = str(Path(__file__).parent / "logging"),
			silent: bool = True,
			verbose: Optional[bool] = True
			) -> Dict[str, List[Dict[str, Any]]]:
		"""
		TODO DOC

		:param dataset:
		:param val_dataset:
		:param logging_path:
		:param silent:
		:param verbose:
		:return:
		"""

		if verbose is not None:
			warnings.warn("The verbose parameter is not supported and will be ignored.")

		self._cfg.work_dir = logging_path

		dataset.pipeline = self._cfg.train_pipeline
		dataloaders = [build_dataloader(
			dataset.get_mmdet_dataset(),
			self.batch_size,
			self.num_workers,
			self._cfg.gpus,
			dist=False,
			seed=self._cfg.seed
		)]

		# Put model on GPUs
		self.model = MMDataParallel(self.model, device_ids=range(self._cfg.gpus)).cuda()

		optimizer = build_optimizer(self.model, self._cfg.optimizer)
		if silent:
			logger = get_root_logger(log_level=logging.WARN)
		else:
			logger = get_root_logger(log_level=logging.INFO)

		# Record some important information such as environment info and seed
		env_info_dict = collect_env()
		env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
		meta = {'env_info': env_info, 'seed': self._cfg.seed}

		runner = Runner(
			self.model,
			batch_processor,
			optimizer,
			self._cfg.work_dir,
			logger=logger,
			meta=meta
		)

		timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
		runner.timestamp = timestamp
		runner.register_training_hooks(self._cfg.lr_config, self._cfg.optimizer_config,
									   self._cfg.checkpoint_config, self._cfg.log_config)

		if val_dataset is not None:
			val_dataset.pipeline = self._cfg.test_pipeline
			val_dataloader = build_dataloader(
				val_dataset.get_mmdet_dataset(test_mode=True),
				imgs_per_gpu=1,
				workers_per_gpu=self.num_workers,
				dist=False,
				shuffle=False
            )
			runner.register_hook(EvalHook(val_dataloader, interval=1, metric=['panoptic']))

		runner.run(dataloaders, self._cfg.workflow, self.iters)
		self._is_model_trained = True

		# Load training statistics from file dumped by the logger
		results = {'train': []}
		if val_dataset is not None:
			results['val'] = []
		for hook in runner.hooks:
			if isinstance(hook, TextLoggerHook):
				with open(hook.json_log_path, 'r') as f:
					for line in f:
						stats = json.loads(line)
						if 'mode' in stats:
							mode = stats.pop('mode', None)
							results[mode].append(stats)
				break

		return results

	def eval(self,
			 dataset: Union[SemanticKittiDataset, NuscenesDataset],
			 print_results: bool = False
			 ) -> Dict[str, Any]:
		raise NotImplementedError

	def infer(self,
			  batch: Union[PointCloud, List[PointCloud]],
			  return_raw_logits: bool = False
			  ):  # TODO: Return Type
		raise NotImplementedError

	def save(self, path):
		raise NotImplementedError

	def load(self, path):
		raise NotImplementedError

	def optimize(self, target_device):
		raise NotImplementedError("EfficientLPS does not need an optimize() method.")

	def reset(self):
		raise NotImplementedError("EfficientLPS is stateless, no reset() needed.")

	@property
	def config(self) -> dict:
		"""
		Getter of internal configurations required by the mmdet API.

		:return: mmdet configuration
		:rtype: dict
		"""
		return self._cfg

	@property
	def num_workers(self) -> int:
		"""
		Getter of number of workers used in the data loaders.

		:return: Number of workers
		:rtype: int
		"""
		return self._num_workers

	@num_workers.setter
	def num_workers(self, value: int):
		"""
		Setter for number of workers used in the data loaders. This will perform the necessary type and value checking.

		:param value: Number of workers
		:type value: int
		"""
		if not isinstance(value, int):
			raise TypeError('num_workers should be an integer.')
		if value <= 0:
			raise ValueError('num_workers should be positive.')
		self._num_workers = value
