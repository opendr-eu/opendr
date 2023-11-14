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

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image as PilImage
from pathlib import Path
from zipfile import ZipFile
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

from mmdet2.apis import single_gpu_test
from mmdet2.apis.train import batch_processor
from mmdet2.core import get_classes, build_optimizer, EvalHook
from mmdet2.datasets import build_dataloader
from mmdet2.datasets.semantic_kitti import STUFF_START_ID
from mmdet2.datasets.pipelines import Compose
from mmdet2.datasets.laserscan_unfolding import LaserScan
from mmdet2.models import build_detector
from mmdet2.utils import collect_env, get_root_logger
from mmdet2.datasets.cityscapes import PALETTE as PALETTE2

from opendr.perception.panoptic_segmentation.datasets import SemanticKittiDataset
from opendr.perception.panoptic_segmentation.datasets.semantic_kitti import PALETTE


Prediction = Union[Tuple[Heatmap, Heatmap, Image], Tuple[np.ndarray, np.ndarray, None]]


class EfficientLpsLearner(Learner):
    """
    The EfficientLpsLearner class provides the top-level API to training and evaluating the EfficientLPS network.
    It particularly facilitates easy inference on Point Cloud inputs when using pre-trained model weights.
    """

    def __init__(self,
                 config_file: Union[str, Path],
                 lr: float=0.07,
                 iters: int=160,
                 batch_size: int=1,
                 optimizer: str="SGD",
                 lr_schedule: Optional[Dict[str, Any]]=None,
                 momentum: float=0.9,
                 weight_decay: float=0.0001,
                 optimizer_config: Optional[Dict[str, Any]]=None,
                 checkpoint_after_iter: int=1,
                 temp_path: Union[str, Path]=Path(__file__).parent / "eval_tmp_dir",
                 device: str="cuda:0",
                 num_workers: int=1,
                 seed: Optional[float]=None,
                 ):
        """
        Constructor

        :param lr: Learning Rate [Training]
        :type lr: float
        :param iters: Number of iterations [Training]
        :type iters: int
        :param batch_size: Size of batches [Training | Evaluation]
        :type batch_size: int
        :param optimizer: Type of the used optimizer [Training]
        :type optimizer: str
        :param lr_schedule: Further settings for the Learning Rate [Training]
        :type lr_schedule: dict
        :param momentum: Momentum used by the optimizer [Training]
        :type momentum: float
        :param weight_decay: Weight decay used by the optimizer [Training]
        :type weight_decay: float
        :param optimizer_config: Further settings for the Optimizer [Training]
        :type optimizer_config: dict
        :param checkpoint_after_iter: Interval in epochs after which to save model checkpoints [Training]
        :type checkpoint_after_iter: int
        :param temp_path: Path to a temporary folder that will be created to evaluate the model [Training | Evaluation]
        :type temp_path: str | Path
        :param device: Hardware device to deploy the model to
        :type device: str
        :param num_workers: Number of workers used by the data loaders [Training | Evaluation]
        :type num_workers: int
        :param seed: Random seed to shuffle the data during training [Training]
        :type seed: float (Optional)
        :param config_file: Path to a python configuration file that contains the model and data loading pipelines.
        :type config_file: str | Path
        """

        super().__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer, temp_path=str(temp_path),
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

        self._cfg = Config.fromfile(str(config_file))
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
        # Redirect stdout to /dev/null for less verbosity
        prev_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self.model = build_detector(self._cfg.model, train_cfg=self._cfg.train_cfg, test_cfg=self._cfg.test_cfg)
        if not hasattr(self.model, "CLASSES"):
            warnings.warn(
                "Class names are not saved in the checkpoint\"s meta data, use Cityscapes classes by default.")
            self.model.CLASSES = get_classes("cityscapes")
        sys.stdout.close()
        sys.stdout = prev_stdout
        self.model.to(self.device)
        self._is_model_trained = False

    def __del__(self):
        """
        Destructor. Deletes the temporary directory.
        """
        shutil.rmtree(self.temp_path, ignore_errors=True)

    def fit(self,
            dataset: SemanticKittiDataset,
            val_dataset: Optional[SemanticKittiDataset]=None,
            logging_path: Union[str, Path]=Path(__file__).parent / "logging",
            silent: bool=True,
            verbose: Optional[bool]=True
            ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Method used for training the model on a train dataset and validating on a separate dataset.

        :param dataset: Object that holds the Training Dataset
        :type dataset: OpenDR Dataset class type
        :param val_dataset: Object that holds the Validation Dataset
        :type val_dataset: OpenDR Dataset class type
        :param logging_path: Path to store the log files, e.g., training progress and tensorboard logs
        :type logging_path: str | Path
        :param silent: If True, printing the training progress to STDOUT is disabled. Evaluation will still be shown.
        :type silent: bool
        :param verbose: Unused.
        :type verbose: bool (Optional)

        :return: Dictionary with "train" and "val" keys containing the training progress (e.g. losses) and,
            if a val_dataset is provided, the evaluation results.
        :rtype: dict
        """

        if verbose is not None:
            warnings.warn("The verbose parameter is not supported and will be ignored.")

        self._cfg.work_dir = str(logging_path)

        dataset.pipeline = self._cfg.train_pipeline
        dataloaders = [build_dataloader(
            dataset.get_mmdet2_dataset(),
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
                val_dataset.get_mmdet2_dataset(test_mode=True),
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
             dataset: SemanticKittiDataset,
             print_results: bool = False
             ) -> Dict[str, Any]:
        """
        Method for evaluating the model on a dataset and returning the following stats:
            - Panoptic Quality (PQ)
            - Segmentation Quality (SQ)
            - Recognition Quality (RQ)
            - Intersection over Union (IoU)

        :param dataset: Object that holds the evaluation dataset
        :type dataset: OpenDR Dataset class type
        :param print_results: Computed metrics will be output to STDOUT if set to True.
        :type print_results: bool

        :return: Dictionary containing the stats regarding the evaluation
        :rtype: dict
        """

        dataset.pipeline = self._cfg.test_pipeline
        dataloader = build_dataloader(
            dataset.get_mmdet2_dataset(test_mode=True),
            imgs_per_gpu=1,
            workers_per_gpu=self.num_workers,
            dist=False,
            shuffle=False
        )

        # Put model on GPUs
        self.model = MMDataParallel(self.model, device_ids=range(self._cfg.gpus)).cuda()

        # Run evaluation
        print("Running inference on dataset...")
        single_gpu_test(self.model, dataloader, show=False, eval=['panoptic'])

        # Move produced output files to a configurable location, since it is hard-coded in the base code.
        std_temp_path = str(Path('tmpDir').absolute())
        if self.temp_path != std_temp_path:
            Path(self.temp_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(std_temp_path, self.temp_path)

        print("Evaluating predictions...")
        results = dataset.evaluate(self.temp_path)

        if print_results:
            msg = f"\n{'Category':<14s}| {'PQ':>5s} {'SQ':>5s} {'RQ':>5s} {'IoU':>5s}\n"
            msg += "-" * 41 + "\n"
            for x in ["all", "things", "stuff"]:
                msg += f"{x:<14s}| {results[x]['PQ'] * 100:>5.1f} {results[x]['SQ'] * 100:>5.1f} "
                msg += f"{results[x]['RQ'] * 100:>5.1f} {results[x]['IoU']:>5.1f}\n"
            msg += "-" * 41 + "\n"
            for cat, value in results['per_class'].items():
                msg += f"{cat:<14s}| {value['PQ'] * 100:>5.1f} {value['SQ'] * 100:>5.1f} "
                msg += f"{value['RQ'] * 100:>5.1f} {value['IoU']:>5.1f}\n"
            msg = msg[:-1]
            print(msg)

        return results

    def pcl_to_mmdet(self,
                     point_cloud: PointCloud,
                     frame_id: int = 0
                     ) -> dict:
        """
        Method for converting an OpenDR PointCloud object to an MMDetector compatible one.
        To be used when the Loading portion of the model's data pipeline is not being used.

        :param point_cloud: Pointcloud to be converted
        :type point_cloud: OpenDR PointCloud
        :param frame_id: Number of the scan frame to be used as its filename. Inferences will use the same filename.
        : type frame_id: int

        :return: An MMDetector compatible dictionary containing the PointCloud data and some additional metadata.
        :rtype: dict
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

        mmdet_pcl = {
            "filename": "",
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

            "stuff_id":  STUFF_START_ID
        }

        return mmdet_pcl

    def infer(self,
              batch: Union[PointCloud, List[PointCloud]],
              return_raw_logits: bool = False,
              projected: bool = False
              ) -> Union[Prediction, List[Prediction]]:
        """
        Method to perform inference on a provided batch of data.

        :param batch: Object that holds a batch of data to run inference on.
        :type batch: Single instance or a list of OpenDR PointCloud objects.
        :param return_raw_logits: Whether the output should be transformed into the OpenDR target class.
        :type return_raw_logits: bool
        :param projected: If True, output will be returned as 2D heatmaps of the spherical projections of the semantic
            and instance labels, as well as the spherical projection of the scan's range.
            Otherwise, the semantic and instance labels will be returned as Numpy arrays.

        :return: A list of predicted targets.
        :rtype: list of tuples of Heatmap class type or list of Numpy arrays
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
            mmdet_img = self.pcl_to_mmdet(point_cloud, frame_id=i)
            mmdet_img = test_pipeline(mmdet_img)
            mmdet_batch.append(scatter(collate([mmdet_img], samples_per_gpu=1), [device])[0])

        results = []
        with torch.no_grad():
            for data in mmdet_batch:
                data['eval'] = 'panoptic'
                if not projected:
                    prediction = self.model(return_loss=False, rescale=True, return_pred=True, **data)[0]
                else:
                    prediction = self.model(return_loss=False, rescale=True, return_pred=False, **data)[0]

                if return_raw_logits:
                    results.append(prediction)
                else:

                    if projected:
                        instance_pred, category_pred, _ = prediction
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
                        panoptic_labels, _ = prediction
                        instance_pred = panoptic_labels >> 16
                        semantic_pred = panoptic_labels & 0xFFFF
                        results.append((instance_pred, semantic_pred, None))

        if single_image_mode:
            return results[0]
        return results

    def save(self,
             path: Union[str, Path]
             ) -> bool:
        """
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str | Path

        :return: Whether save succeeded or not
        :rtype: bool
        """

        if not self._is_model_trained:
            warnings.warn("The current model has not been trained.")

        # Create structure according to OpenDR specification
        dir_path = path
        if not isinstance(path, Path):
            dir_path = Path(path)
        dir_path = dir_path / "efficient_lps"
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
            "optimizer_info": {},
            "classes": self.model.CLASSES,
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

    def load(self,
             path: Union[str, Path]
             ) -> bool:
        """
        Loads a model from the provided path.

        :param path: Path to saved model
        :type path: str | Path

        :return: Whether load succeeded or not
        :rtype: bool
        """

        if not isinstance(path, Path):
            path = Path(path)

        if path.suffix == ".pth":  # Read the actual model
            try:
                checkpoint = load_checkpoint(self.model, str(path))
                self.model.CLASSES = checkpoint["meta"].get("CLASSES")
                if not self.model.CLASSES:
                    warnings.warn(
                        "Class names are not saved in the checkpoint\"s meta data, use Cityscapes classes by default.")
                    self.model.CLASSES = get_classes("cityscapes")
                self._is_model_trained = True
            except (RuntimeError, OSError):
                return False
            return True
        else:  # OpenDR specification
            meta_path = path / f"{path.name}.json"
            if not meta_path.exists():
                warnings.warn(f"No model meta data found at {meta_path}")
                return False
            with open(meta_path, "r") as f:
                meta_data = json.load(f)
            # According to the OpenDR specification, the model path is given with a leading slash
            model_path = path / str(meta_data["model_paths"]).lstrip("/")
            if not model_path.exists():
                warnings.warn(f"No model weights found at {model_path}")
                return False
            return self.load(model_path)

    def optimize(self, target_device: str) -> bool:
        # Not needed for this learner.
        raise NotImplementedError("EfficientLPS does not need an optimize() method.")

    def reset(self) -> None:
        # Not needed for this learner since it is stateless.
        raise NotImplementedError("EfficientLPS is stateless, no reset() needed.")

    @staticmethod
    def download(path: Union[str, Path],
                 mode: str = "model",
                 trained_on: str = "semantickitti",
                 prepare_data: bool = False,
                 ) -> str:
        """
        Download data from the OpenDR server.
        Valid modes include pre-trained model weights and data used in the unit tests.

        Currently, the following pre-trained models are available:
            - SemanticKITTI panoptic segmentation dataset

        :param path: Path to save the model weights
        :type path: str | Path
        :param mode: What kind of data to download
        :type mode: str
        :param trained_on: Dataset on which the model has been trained [applicable only to mode == "model"]
        :type trained_on: str

        :return: Path to the downloaded file
        :rtype: str
        """
        if mode == "model":
            models = {
                "semantickitti":
                f"{OPENDR_SERVER_URL}perception/panoptic_segmentation/efficient_lps/models/model_semantickitti.pth"
            }
            if trained_on not in models.keys():
                raise ValueError(f"Could not find model weights pre-trained on {trained_on}. "
                                 f"Valid options are {list(models.keys())}")
            url = models[trained_on]
        elif mode == "test_data":
            url = f"{OPENDR_SERVER_URL}perception/panoptic_segmentation/efficient_lps/test_data.zip"
        else:
            raise ValueError("Invalid mode. Valid options are ['model', 'test_data']")

        if not isinstance(path, Path):
            path = Path(path)
        filename = path / url.split("/")[-1]
        path.mkdir(parents=True, exist_ok=True)

        def pbar_hook(prog_bar: tqdm):
            prev_b = [0]

            def update_to(b=1, bsize=1, total=None):
                if total is not None:
                    prog_bar.total = total
                prog_bar.update((b - prev_b[0]) * bsize)
                prev_b[0] = b

            return update_to

        if os.path.exists(filename) and os.path.isfile(filename):
            print(f'File already downloaded: {filename}')
        else:
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {filename}")\
                 as pbar:
                urllib.request.urlretrieve(url, filename, pbar_hook(pbar))
        if prepare_data and mode == "test_data":
            print(f"Extracting {filename}")
            try:
                with ZipFile(filename, 'r') as zipObj:
                    zipObj.extractall(path)
                os.remove(filename)
            except:
                print(f"Could not extract {filename} to {path}. Please extract it manually.")
                print("The data might have been already extracted an is available in the test_data folder.")
            path = os.path.join(path, "test_data", "eval_data")
            return path

        return str(filename)

    @staticmethod
    def visualize(pointcloud: PointCloud,
                  predictions: Tuple[np.ndarray, np.ndarray],
                  show_figure: bool = True,
                  save_figure: bool = False,
                  figure_filename: Optional[Union[str, Path]]=None,
                  figure_size: Tuple[float, float]=(15, 10),
                  max_inst: int=20,
                  min_alpha: float=0.25,
                  dpi: int=600,
                  return_pointcloud: Optional[bool]=False,
                  return_pointcloud_type: Optional[str]=None,
                  ) -> Union[PointCloud, Image]:
        """
        Create a visualization of the predicted panoptic segmentation as a colored 3D Scatter Plot.

        :param pointcloud: Pointcloud object containing the coordinates of the scan
        :type pointcloud: OpenDR PointCloud object
        :param predictions: Tuple of semantic and instance labels
        :type predictions: tuple of Numpy arrays
        :param show_figure: Displays the resulting plot in a GUI if True
        :type show_figure: bool
        :param save_figure: Saves the resulting plot to a *.png file if True
        :type save_figure: bool
        :param figure_filename: Name of the image to be saved. Required if save_figure is set to True
        :type figure_filename: str
        :param figure_size: Size of the figure in inches. Wrapper of matplotlib figuresize.
        :type figure_size: Tuple of floats
        :param max_inst: Maximum value that the instance ID can take. Used for computing the alpha value of a point.
        :type max_inst: int
        :param min_alpha: Minimum value that a point's alpha value can take, so that it is never fully transparent.
        :type min_alpha: float
        :param dpi: Resolution of the resulting image, in Dots per Inch.
        :type dpi: int
        :param return_pointcloud: If True, returns a PointCloud object with the predicted labels as colors.
        :type return_pointcloud: bool
        :param return_pointcloud_type: If return_pointcloud is True, this parameter specifies the type of the returned
            PointCloud object. Valid options are "semantic", "instance" and "panoptic".
        :type return_pointcloud_type: str

        :return: OpenDR Image of the generated visualization or OpenDR PointCloud with the predicted labels
        :rtype: OpenDR Image or OpenDR PointCloud
        """

        if save_figure and figure_filename is None:
            raise ValueError("Argument figure_filename must be specified if save_figure is True.")

        points = pointcloud.data
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        inst = predictions[0]
        sem = predictions[1]

        if return_pointcloud:  # Return the pointcloud with panoptic labels
            colors = np.array(PALETTE, dtype=np.uint8)
            sem[sem == 255] = 19  # Convert unlabelled points into the last index of the palette
            colors = colors[sem]  # Get the colors for each semantic label
            if return_pointcloud_type == "panoptic":
                inst[sem >= 8] = -1  # Set the instance labels of the stuff classes to -1
                if np.unique(inst).shape[0] > 1:
                    n_inst = np.unique(inst)[-2]  # Get the number of instances
                    # Generate a new palette for the instances
                    new_palette = np.random.randint(0, 255, size=(n_inst, 3), dtype=np.uint8)
                    for i in range(new_palette.shape[0]):  # Assign new color into instances
                        color = new_palette[i]  # Get the color from new PALETTE
                        while color in PALETTE:  # Make sure that the new color is not in the semantic palette
                            color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8)
                        colors[inst == (i+1)] = color  # Assign the new color to the instance
                points = np.c_[x, y, z, colors]  # Concatenate the points with the colors

            elif return_pointcloud_type == "semantic":
                points = np.c_[x, y, z, colors]  # Concatenate the points with the colors

            elif return_pointcloud_type == "instance":
                colors[sem >= 8] = np.array([0, 0, 0]).reshape(1, 3)  # Set the colors of the stuff classes to black
                inst[sem >= 8] = -1
                n_inst = np.unique(inst)[-2]  # Get the number of instances
                # Generate a new palette for the instances
                new_palette = np.random.randint(0, 255, size=(n_inst, 3), dtype=np.uint8)
                for i in range(new_palette.shape[0]):  # Assign new color into instances
                    color = new_palette[i]  # Get the color from new PALETTE
                    if np.array_equal(color, np.array([0, 0, 0]).reshape(1, 3)):  # Make sure that the new color is not black
                        color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8)
                    colors[inst == (i+1)] = color  # Assign the new color to the instance
                points = np.c_[x, y, z, colors]  # Concatenate the points with the colors

            else:
                raise ValueError("Argument return_pointcloud_type must be one of 'panoptic', 'semantic', or 'instance'.")

            pc = PointCloud(points)  # Create RGBXYZ pointcloud input
            return pc

        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))  # Set aspect ratio to 1:1:1 in data space

        PALETTE2.append([0, 0, 0])
        colors = np.array(PALETTE2, dtype=np.float) / 255.
        colors = colors[sem % colors.shape[0]]  # Use the mod of the sem. label in case some values aren't in the colors
        alphas = ((min_alpha - 1) / max_inst) * inst + 1
        alphas = np.clip(alphas, min_alpha, 1)
        colors = np.c_[colors, alphas]

        ax.scatter(x, y, z, s=0.25, c=colors)  # Plot the point cloud

        fig.canvas.draw()
        visualization_img = PilImage.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()

        if save_figure:
            if not isinstance(figure_filename, Path):
                figure_filename = Path(figure_filename)
            figure_filename.parent.mkdir(parents=True, exist_ok=True)
            visualization_img.save(figure_filename)
        if show_figure:
            visualization_img.show()

        # Explicitly convert from HWC/RGB (PIL) to CHW/RGB (OpenDR)
        return Image(data=np.array(visualization_img).transpose((2, 0, 1)), guess_format=False)

    @property
    def config(self) -> dict:
        """
        Getter of internal configurations required by the mmdet2 API.

        :return: mmdet2 configuration
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
