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

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image as PilImage
from pathlib import Path
import shutil
import sys
import time
import torch
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Union, Tuple
import urllib
import warnings

from mmcv import Config
from mmcv.parallel import scatter, collate, MMDataParallel
from mmcv.runner import load_checkpoint, save_checkpoint, Runner, TextLoggerHook

from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.data import PointCloud, Image
from opendr.engine.learners import Learner
from opendr.engine.target import Heatmap

from mmdet.apis import single_gpu_test
from mmdet.apis.train import batch_processor
from mmdet.core import get_classes, build_optimizer, EvalHook
from mmdet.datasets import build_dataloader
from mmdet.datasets.cityscapes import PALETTE
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.laserscan_unfolding import LaserScan
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from opendr.perception.panoptic_segmentation.datasets import SemanticKittiDataset, NuscenesDataset


class EfficientLpsLearner(Learner):
	"""
	The EfficientLpsLearner class provides the top-level API to training and evaluating the EfficientLPS network.
	It particularly facilitates easy inference on Point Cloud inputs when using pre-trained model weights.
	"""

	def __init__(self,
				 lr: float = 0.07,
				 iters: int = 160,
				 batch_size: int = 1,
				 optimizer: str = "SGD",
				 lr_schedule: Optional[Dict[str, Any]] = None,
				 momentum: float = 0.9,
				 weight_decay: float = 0.0001,
				 optimizer_config: Optional[Dict[str, Any]] = None,
				 checkpoint_after_iter: int = 1,
				 temp_path: str = str(Path(__file__).parent / "eval_tmp_dir"),
				 device: str = "cuda:0",
				 num_workers: int = 1,
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
				"policy": "step",
				"warmup": "linear",
				"warmup_iters": 500,
				"warmup_ratio": 1 / 3,
				"step": [120, 144]
			}

		self._lr_schedule = lr_schedule

		if optimizer_config is None:
			optimizer_config = {
				"grad_clip": {
					"max_norm": 35,
					"norm_type": 2
				}
			}

		self._checkpoint_after_iter = checkpoint_after_iter
		self._num_workers = num_workers

		self._cfg = Config.fromfile(config_file)
		self._cfg.workflow = [("train", 1)]
		self._cfg.model.pretrained = None
		self._cfg.optimizer = {
			"type": self.optimizer,
			"lr": self.lr,
			"momentum": momentum,
			"weight_decay": weight_decay
		}
		self._cfg.optimizer_config = optimizer_config
		self._cfg.lr_config = self.lr_schedule
		self._cfg.total_epochs = self.iters
		self._cfg.checkpoint_config = {"interval": self.checkpoint_after_iter}
		self._cfg.log_config = {
			"interval": 1,
			"hooks": [
				{"type": "TextLoggerHook"},
				{"type": "TensorboardLoggerHook"}
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
		meta = {"env_info": env_info, "seed": self._cfg.seed}

		runner = Runner(
			self.model,
			batch_processor,
			optimizer,
			self._cfg.work_dir,
			logger=logger,
			meta=meta
		)

		timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
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
			runner.register_hook(EvalHook(val_dataloader, interval=1, metric=["panoptic"]))

		runner.run(dataloaders, self._cfg.workflow, self.iters)
		self._is_model_trained = True

		# Load training statistics from file dumped by the logger
		results = {"train": []}
		if val_dataset is not None:
			results["val"] = []
		for hook in runner.hooks:
			if isinstance(hook, TextLoggerHook):
				with open(hook.json_log_path, "r") as f:
					for line in f:
						stats = json.loads(line)
						if "mode" in stats:
							mode = stats.pop("mode", None)
							results[mode].append(stats)
				break

		return results

	def eval(self,
			 dataset: Union[SemanticKittiDataset, NuscenesDataset],
			 print_results: bool = False
			 ) -> Dict[str, Any]:
		"""
		TODO:

		:param dataset:
		:type dataset:
		:param print_results:
		:type print_results:
		:return:
		"""

		dataset.pipeline = self._cfg.test_pipeline
		dataloader = build_dataloader(
			dataset.get_mmdet_dataset(test_mode=True),
			imgs_per_gpu=1,
			workers_per_gpu=self.num_workers,
			dist=False,
			shuffle=False
		)

		# Put model on GPUs
		self.model = MMDataParallel(self.model, device_ids=range(self._cfg.gpus)).cuda()

		# Run evaluation
		single_gpu_test(self.model, dataloader, show=False, eval=['panoptic'])
		std_temp_path = Path('tmpDir').absolute()  # This is hard-coded in the base code
		if self.temp_path != std_temp_path:
			shutil.copytree(std_temp_path, self.temp_path, dirs_exist_ok=True)
			shutil.rmtree(std_temp_path)

		prev_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')  # Block prints to STDOUT
		results = dataset.evaluate(os.path.join(self.temp_path, 'tmp'), os.path.join(self.temp_path, 'tmp_json'))
		sys.stdout.close()
		sys.stdout = prev_stdout

		if print_results:
			msg = f"\n{'Category':<14s}| {'PQ':>5s} {'SQ':>5s} {'RQ':>5s} {'N':>5s}\n"
			msg += "-" * 41 + "\n"
			for x in ['All', 'Things', 'Stuff']:
				msg += f"{x:<14s}| {results[x]['pq'] * 100:>5.1f} {results[x]['sq'] * 100:>5.1f} "
				msg += f"{results[x]['rq'] * 100:>5.1f} {results[x]['n']:>5d}\n"
			msg += "-" * 41 + "\n"
			for cat, value in results['per_class'].items():
				msg += f"{cat:<14s}| {value['pq'] * 100:>5.1f} {value['sq'] * 100:>5.1f} {value['rq'] * 100:>5.1f}\n"
			msg = msg[:-1]
			print(msg)

		return results

	def pcl_to_mmet(self, point_cloud: PointCloud, file_count: int = 0) -> dict:
		"""
		TODO
		:param point_cloud:
		:param file_count:
		:return:
		:rtype
		"""

		loader_pipeline_cfg = self._cfg.test_pipeline[0]
		project = loader_pipeline_cfg["project"]
		h = loader_pipeline_cfg["H"]
		w = loader_pipeline_cfg["W"]
		fov_up = loader_pipeline_cfg["fov_up"]
		fov_down = loader_pipeline_cfg["fov_down"]
		max_points = loader_pipeline_cfg["max_points"]
		sensor_img_means = loader_pipeline_cfg["sensor_img_means"]
		sensor_img_stds = loader_pipeline_cfg["sensor_img_means"]

		scan = LaserScan(project=project, H=h, W=w, fov_up=fov_up, fov_down=fov_down)
		remissions = None
		shape = point_cloud.data.shape
		if shape[-1] < 3:
			raise ValueError("Invalid point cloud shape")
		elif shape[-1] == 3:
			points = point_cloud.data
		else:
			points = point_cloud.data[:, :3]
			remissions = point_cloud.data[:, 3]

		scan.set_points(points, remissions=remissions)
		scan.do_range_projection()

		unproj_n_points = scan.points.shape[0]
		unproj_xyz = torch.full((max_points, 3), -1.0, dtype=torch.float)
		unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
		unproj_range = torch.full([max_points], -1.0, dtype=torch.float)
		unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
		unproj_remissions = torch.full([max_points], -1.0, dtype=torch.float)
		unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

		sensor_img_means = torch.tensor(sensor_img_means,
										dtype=torch.float).view(-1, 1, 1)
		sensor_img_stds = torch.tensor(sensor_img_stds,
									   dtype=torch.float).view(-1, 1, 1)

		# get points and labels
		proj_range = torch.from_numpy(scan.proj_range).clone()
		proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
		proj_remission = torch.from_numpy(scan.proj_remission).clone()
		proj_mask = torch.from_numpy(scan.proj_mask)
		proj_x = torch.full([max_points], -1, dtype=torch.long)
		proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
		proj_y = torch.full([max_points], -1, dtype=torch.long)
		proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
		proj = torch.cat([proj_range.unsqueeze(0).clone(),
						  proj_xyz.clone().permute(2, 0, 1),
						  proj_remission.unsqueeze(0).clone()])
		proj = (proj - sensor_img_means
				) / sensor_img_stds
		proj = proj * proj_mask.float()

		results = {
			# TODO: Awful Hack!
			# Prediction will be saved in $PWD/tmpDir/inference/predictions/pcl_<file_count>.label
			#                                         |||||||||             \\\\\\\\\\\\\\\\\
			#                                         VVVVVVVVV              VVVVVVVVVVVVVVVVV
			"filename": f"expected_path_for_something/inference/__dont_care__/pcl_{file_count:06d}.bin",
			"img": proj,
			"img_shape": proj.shape,
			"ori_shape": proj.shape,

			"pad_shape": proj.shape,
			"scale_factor": 1.0,

			"proj_x": proj_x,
			"proj_y": proj_y,
			"proj_msk": proj_mask,
			"proj_range": proj_range,
			"unproj_range": unproj_range,
			"unproj_n_points": unproj_n_points,
			"sensor_img_means": sensor_img_means,
			"sensor_img_stds": sensor_img_stds,

			"stuff_id":  0  # TODO: Check if this doesn't break anything!!!!
		}

		return results

	def infer(self,
			  batch: Union[PointCloud, List[PointCloud]],
			  return_raw_logits: bool = False,
			  projected: bool = False
			  ) -> List[Tuple[np.ndarray, np.ndarray, Optional[Image]]]:
		"""
		TODO
		:param batch:
		:param return_raw_logits:
		:param projected:
		:return:
		"""

		if self.model is None:
			raise RuntimeError("No model loaded.")
		if not self._is_model_trained:
			warnings.warn("The current model has not been trained.")
		self.model.eval()

		# Build the data pipeline
		test_pipeline = Compose(self._cfg.test_pipeline[1:])
		device = next(self.model.parameters()).device

		# Convert to the format expected by the mmdetection API
		single_image_mode = False
		if isinstance(batch, PointCloud):
			batch = [batch]
			single_image_mode = True

		mmdet_batch = []
		for i, point_cloud in enumerate(batch):
			mmdet_img = self.pcl_to_mmet(point_cloud, file_count=i)
			mmdet_img = test_pipeline(mmdet_img)
			mmdet_batch.append(scatter(collate([mmdet_img], samples_per_gpu=1), [device])[0])

		results = []
		with torch.no_grad():
			for data in mmdet_batch:
				data['eval'] = 'panoptic'
				prediction = self.model(return_loss=False, rescale=True, **data)[0]

				if return_raw_logits:
					results.append(prediction)
				else:
					instance_pred, category_pred, img_meta = prediction

					if projected:
						instance_pred = instance_pred.numpy()
						semantic_pred = category_pred[instance_pred].numpy()

						instance_pred = Heatmap(instance_pred.astype(np.uint8))
						semantic_pred = Heatmap(semantic_pred.astype(np.uint8))

						ranges = data["img"][0, 0, :, :].cpu().numpy()  # Depth map (Range)
						ranges = np.clip(ranges * 255 / ranges.max(), 0, 255).astype(np.uint8)
						ranges = np.repeat(ranges[np.newaxis, :, :], 3, axis=0)
						ranges = Image(ranges.astype(np.uint8))

						results.append((instance_pred, semantic_pred, ranges))
					else:
						# Load the predicted labels from the saved file, since the un-projected labels are not returned by
						panoptic_labels_path = self.model.get_path(img_meta)
						panoptic_labels = np.fromfile(panoptic_labels_path, dtype=np.uint32)
						instance_pred = panoptic_labels >> 16
						semantic_pred = panoptic_labels & 0xFFFF
						results.append((instance_pred, semantic_pred, None))

		if single_image_mode:
			return results[0]
		return results

	def save(self, path):
		"""
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str
        :return: Whether save succeeded or not
        :rtype: bool
        """
		if not self._is_model_trained:
			warnings.warn("The current model has not been trained.")
	
		# Create structure according to OpenDR specification
		dir_path = Path(path) / "efficient_lps"
		if dir_path.exists():
			warnings.warn("The given path already exists. Any content will be overwritten.")
		else:
			dir_path.mkdir(parents=True)
		model_path = dir_path / "model.pth"
		meta_path = dir_path / "efficient_lps.json"
	
		meta_data = {
			"model_paths": [f"/{model_path.parent.name}/{model_path.name}"],
			"framework": "pytorch",
			"format": "pth",
			"has_data": False,
			"inference_params": {},
			"optimized": self._is_model_trained,
			"optimizer_info": {}
		}
	
		with open(meta_path, "w") as f:
			json.dump(meta_data, f, indent=True, sort_keys=True)
	
		try:
			# Save the actual model
			save_checkpoint(self.model, str(model_path))
		except TypeError:
			return False
		if not model_path.exists() or not meta_path.exists():
			return False
		return True

	def load(self, path):
		"""
        Loads a model from the path provided.

        :param path: Path to saved model
        :type path: str
        :return: Whether load succeeded or not
        :rtype: bool
        """
		if ".pth" in path:  # Read the actual model
			try:
				checkpoint = load_checkpoint(self.model, path)
				if "CLASSES" in checkpoint["meta"]:
					self.model.CLASSES = checkpoint["meta"]["CLASSES"]
				else:
					warnings.warn(
						"Class names are not saved in the checkpoint\"s meta data, use Cityscapes classes by default.")
					self.model.CLASSES = get_classes("cityscapes")
				self._is_model_trained = True
			except (RuntimeError, OSError):
				return False
			return True
		else:  # OpenDR specification
			meta_path = Path(path) / f"{Path(path).name}.json"
			if not meta_path.exists():
				warnings.warn(f"No model meta data found at {meta_path}")
				return False
			with open(meta_path, "r") as f:
				meta_data = json.load(f)
			# According to the OpenDR specification, the model path is given with a leading slash
			model_path = Path(path) / str(meta_data["model_paths"]).lstrip("/")
			if not model_path.exists():
				warnings.warn(f"No model weights found at {model_path}")
				return False
			return self.load(str(model_path))

	def optimize(self, target_device):
		# Not needed for this learner.
		raise NotImplementedError("EfficientLPS does not need an optimize() method.")

	def reset(self):
		# Not needed for this learner since it is stateless.
		raise NotImplementedError("EfficientLPS is stateless, no reset() needed.")

	@staticmethod
	def download(path: str, mode: str = "model", trained_on: str = "kitti") -> str:
		"""
		TODO
		Download data from the OpenDR server. Valid modes include pre-trained model weights and data used in the unit tests.
	
		Currently, the following pre-trained models are available:
			- KITTI panoptic segmentation dataset
			- NuScenes
	
		:param path: Path to save the model weights
		:type path: str
		:param mode: What kind of data to download
		:type mode: str
		:param trained_on: Dataset on which the model has been trained [applicable only to mode == "model"]
		:type trained_on: str
		:return: Path to the downloaded file
		:rtype: str
		"""
		if mode == "model":
			models = {
				# TODO: Check URLs after uploading models
				"nuscenes": f"{OPENDR_SERVER_URL}perception/panoptic_segmentation/efficientlps/models/model_nuscenes.pth",
				"kitti": f"{OPENDR_SERVER_URL}perception/panoptic_segmentation/efficientlps/models/model_semantickitti.pth"
			}
			if trained_on not in models.keys():
				raise ValueError(f"Could not find model weights pre-trained on {trained_on}. "
								 f"Valid options are {list(models.keys())}")
			url = models[trained_on]
		elif mode == "test_data":
			# TODO: Check URLs after uploading models
			url = f"{OPENDR_SERVER_URL}perception/panoptic_segmentation/efficientlps/test_data/test_data.zip"
		else:
			raise ValueError("Invalid mode. Valid options are ['model', 'test_data']")
	
		filename = os.path.join(path, url.split("/")[-1])
		os.makedirs(path, exist_ok=True)
	
		def pbar_hook(prog_bar: tqdm):
			prev_b = [0]
	
			def update_to(b=1, bsize=1, total=None):
				if total is not None:
					prog_bar.total = total
				prog_bar.update((b - prev_b[0]) * bsize)
				prev_b[0] = b
	
			return update_to
	
		with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {filename}") as pbar:
			urllib.request.urlretrieve(url, filename, pbar_hook(pbar))
		return filename

	@staticmethod
	def visualize(pointcloud: PointCloud,
				  predictions: Tuple[np.ndarray, np.ndarray],
				  show_figure: bool = True,
				  save_figure: bool = False,
				  figure_filename: Optional[str] = None,
				  figure_size: Tuple[float, float] = (15, 10),
				  detailed: bool = False,
				  max_inst: int = 20,
				  min_alpha: float = 0.25,
				  dpi: int = 600,
				  ) -> Image:
		"""
		TODO:

		:param pointcloud:
		:param predictions:
		:param show_figure:
		:param save_figure:
		:param figure_filename:
		:param figure_size:
		:param detailed:
		:param max_inst:
		:param min_alpha:
		:param dpi:
		:return:
		"""

		del detailed  # Unused parameter, kept so that signature is like that of EfficientPsLearner.visualize()

		if save_figure and figure_filename is None:
			raise ValueError("Argument figure_filename must be specified if save_figure is True.")

		points = pointcloud.data
		x = points[:, 0]
		y = points[:, 1]
		z = points[:, 2]

		inst = predictions[0]
		sem = predictions[1]

		fig = plt.figure(figsize=figure_size, dpi=dpi)
		ax = fig.add_subplot(111, projection="3d")
		ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))  # Set aspect ratio to 1:1:1 in data space

		PALETTE.append([0, 0, 0])
		colors = np.array(PALETTE, dtype=np.float) / 255.
		colors = colors[sem]
		alphas = ((min_alpha - 1) / max_inst) * inst + 1
		alphas = np.clip(alphas, min_alpha, 1)
		colors = np.c_[colors, alphas]

		ax.scatter(x, y, z, s=0.25, c=colors)  # Plot the point cloud

		fig.canvas.draw()
		visualization_img = PilImage.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
		plt.close()

		if save_figure:
			visualization_img.save(figure_filename)
		if show_figure:
			visualization_img.show()

		# Explicitly convert from HWC/RGB (PIL) to CHW/RGB (OpenDR)
		return Image(data=np.array(visualization_img).transpose((2, 0, 1)), guess_format=False)

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
			raise TypeError("num_workers should be an integer.")
		if value <= 0:
			raise ValueError("num_workers should be positive.")
		self._num_workers = value
