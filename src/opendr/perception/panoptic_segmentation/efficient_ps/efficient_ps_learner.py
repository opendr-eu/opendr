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
import os
import shutil
import time
import urllib
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import scatter, collate, MMDataParallel
from mmcv.runner import load_checkpoint, save_checkpoint, Runner
from mmdet.apis.train import batch_processor
from mmdet.core import get_classes, build_optimizer, EvalHook, save_panoptic_eval
from mmdet.datasets import build_dataloader
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.data import Image
from opendr.engine.learners import Learner
from opendr.engine.target import Heatmap
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset, KittiDataset
from opendr.perception.panoptic_segmentation.datasets import Image as ImageWithFilename


class EfficientPsLearner(Learner):
    """
    The EfficientPsLearner class provides the top-level API to training and evaluating the EfficientPS network.
    Particularly, it facilitates easy inference on RGB images when using pre-trained model weights.
    """

    def __init__(self,
                 lr: float=.07,
                 iters: int=160,
                 batch_size: int=1,
                 optimizer: str='SGD',
                 lr_schedule: Optional[Dict[str, Any]]=None,
                 momentum: float=.9,
                 weight_decay: float=.0001,
                 optimizer_config: Optional[Dict[str, Any]]=None,
                 checkpoint_after_iter: int=1,
                 temp_path: str=str(Path(__file__).parent / 'eval_tmp_dir'),
                 device: str="cuda:0",
                 num_workers: int=1,
                 seed: Optional[float]=None,
                 config_file: str=str(Path(__file__).parent / 'configs' / 'singlegpu_sample.py')
                 ):
        """
        :param lr: learning rate [training]
        :type lr: float
        :param iters: number of iterations [training]
        :type iters: int
        :param batch_size: size of batches [training, evaluation]
        :type batch_size: int
        :param optimizer: type of the utilized optimizer [training]
        :type optimizer: str
        :param lr_schedule: further settings for the learning rate [training]
        :type lr_schedule: dict
        :param momentum: momentum used by the optimizer [training]
        :type momentum: float
        :param weight_decay: weight decay used by the optimizer [training]
        :type weight_decay: float
        :param optimizer_config: further settings for the optimizer [training]
        :type optimizer_config: dict
        :param checkpoint_after_iter: defines the interval in epochs to save checkpoints [training]
        :type checkpoint_after_iter: int
        :param temp_path: path to a temporary folder that will be created to evaluate the model [training, evaluation]
        :type temp_path: str
        :param device: the device to deploy the model
        :type device: str
        :param num_workers: number of workers used by the data loaders [training, evaluation]
        :type num_workers: int
        :param seed: random seed to shuffle the data during training [training]
        :type seed: float, optional
        :param config_file: path to a config file that contains the model and the data loading pipelines
        :type config_file: str
        """
        super().__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer, temp_path=temp_path,
                         device=device)
        if lr_schedule is None:
            lr_schedule = {'policy': 'step', 'warmup': 'linear', 'warmup_iters': 500, 'warmup_ratio': 1 / 3,
                           'step': [120, 144]}
        if optimizer_config is None:
            optimizer_config = {'grad_clip': {'max_norm': 35, 'norm_type': 2}}
        self._lr_schedule = lr_schedule
        self._checkpoint_after_iter = checkpoint_after_iter
        self._num_workers = num_workers

        self._cfg = Config.fromfile(config_file)
        self._cfg.workflow = [('train', 1)]
        self._cfg.model.pretrained = None
        self._cfg.optimizer = {'type': self.optimizer, 'lr': self.lr, 'momentum': momentum,
                               'weight_decay': weight_decay}
        self._cfg.optimizer_config = optimizer_config
        self._cfg.lr_config = self.lr_schedule
        self._cfg.total_epochs = self.iters
        self._cfg.checkpoint_config = {'interval': self.checkpoint_after_iter}
        self._cfg.log_config = {'interval': 1, 'hooks': [{'type': 'TextLoggerHook'}, {'type': 'TensorboardLoggerHook'}]}
        self._cfg.gpus = 1  # Numbers of GPUs to use (only applicable to non-distributed training)
        self._cfg.seed = seed

        # Create model
        self.model = build_detector(self._cfg.model, train_cfg=self._cfg.train_cfg, test_cfg=self._cfg.test_cfg)
        self.model.to(self.device)
        self._is_model_trained = False

    def fit(self,
            dataset: Any,
            val_dataset: Optional[Union[CityscapesDataset, KittiDataset]]=None,
            logging_path: str=str(Path(__file__).parent / 'logging'),
            silent: bool=False,
            verbose: Optional[bool]=None
            ):
        """
        This method is used for training the algorithm on a train dataset and validating on a separate dataset.

        :param dataset: Object that holds the training dataset
        :type dataset: Dataset class type
        :param val_dataset: Object that holds the validation dataset
        :type val_dataset: Dataset class type, optional
        :param logging_path: Path to store the logging files, e.g., training progress and tensorboard logs
        :type logging_path: str
        :param silent: if True, disables printing training progress reports to STDOUT. The evaluation will still be shown.
        :type silent: bool
        """
        if verbose is not None:
            warnings.warn('The verbose parameter is not supported and will be ignored.')

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
        env_info = '\n'.join([('{}: {}'.format(k, v)) for k, v in env_info_dict.items()])
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

    def eval(self,
             dataset: Any,
             print_results: bool=False
             ) -> Dict[str, Any]:
        """
        This method is used to evaluate the algorithm on a dataset and returns the following stats:
            - Panoptic Quality (PQ)
            - Segmentation Quality (SQ)
            - Recognition Quality (RQ)

        :param dataset: Object that holds the evaluation dataset
        :type dataset: Dataset class type
        :param print_results: If set to True, the computed metrics will be printed to STDOUT
        :type print_results: bool
        :return: Returns stats regarding the evaluation
        :rtype: dict
        """
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers,
                                collate_fn=lambda x: x)

        with tqdm(dataloader, unit='batch', desc='Evaluating...') as pbar:
            for i, batch in enumerate(dataloader):
                images = [data[0] for data in batch]
                predictions = self.infer(images, return_raw_logits=True)
                save_panoptic_eval(predictions, path=self.temp_path)
                pbar.update(1)

        results = dataset.evaluate(os.path.join(self.temp_path, 'tmp'), os.path.join(self.temp_path, 'tmp_json'))

        if print_results:
            msg = f"{'Category':<14s}| {'PQ':>5s} {'SQ':>5s} {'RQ':>5s} {'N':>5s}\n"
            msg += "-" * 41 + "\n"
            for x in ['All', 'Things', 'Stuff']:
                msg += f"{x:<14s}| {results[x]['pq'] * 100:>5.1f} {results[x]['sq'] * 100:>5.1f} "
                msg += f"{results[x]['rq'] * 100:>5.1f} {results[x]['n']:>5d}\n"
            msg += "-" * 41 + "\n"
            for cat, value in results['per_class'].items():
                msg += f"{cat:<14s}| {value['pq'] * 100:>5.1f} {value['sq'] * 100:>5.1f} {value['rq'] * 100:>5.1f}\n"
            msg = msg[:-1]
            print(msg)

        shutil.rmtree(self.temp_path)
        return results

    def infer(self,
              batch: Union[Image, List[Image], ImageWithFilename, List[ImageWithFilename]],
              return_raw_logits: bool=False
              ) -> Union[List[Tuple[Heatmap, Heatmap]], Tuple[Heatmap, Heatmap], np.ndarray]:
        """
        This method performs inference on the batch provided.

        :param batch: Object that holds a batch of data to run inference on
        :type batch: Image class type or list of Image class type
        :param return_raw_logits: Whether the output should be transformed into the OpenDR target class.
        :type return_raw_logits: bool
        :return: A list of predicted targets
        :rtype: list of tuples of Heatmap class type or list of numpy arrays
        """
        if self.model is None:
            raise RuntimeError('No model loaded.')
        if not self._is_model_trained:
            warnings.warn('The current model has not been trained.')
        self.model.eval()

        # Build the data pipeline
        test_pipeline = Compose(self._cfg.test_pipeline[1:])
        device = next(self.model.parameters()).device

        # Convert to the format expected by the mmdetection API
        single_image_mode = False
        if isinstance(batch, Image) or isinstance(batch, ImageWithFilename):
            batch = [batch]
            single_image_mode = True
        mmdet_batch = []
        for img in batch:
            if isinstance(img, ImageWithFilename):
                filename = img.filename
            else:
                filename = None
            mmdet_img = {'filename': filename, 'img': img.numpy(), 'img_shape': img.numpy().shape,
                         'ori_shape': img.numpy().shape}
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
                    instance_pred, category_pred, _ = prediction
                    instance_pred = instance_pred.numpy()
                    semantic_pred = category_pred[instance_pred].numpy()

                    # Some pixels have not gotten a semantic class assigned because they are marked as stuff by the
                    # instance head but not by the semantic segmentation head
                    # We mask them as 255 in the semantic segmentation map

                    instance_pred = Heatmap(instance_pred.astype(np.uint8), description='Instance prediction')
                    semantic_pred = Heatmap(semantic_pred.astype(np.uint8), description='Semantic prediction')
                    results.append((instance_pred, semantic_pred))

        if single_image_mode:
            return results[0]
        else:
            return results

    def save(self, path: str) -> bool:
        """
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str
        :return: Whether save succeeded or not
        :rtype: bool
        """
        if not self._is_model_trained:
            warnings.warn('The current model has not been trained.')

        try:
            save_checkpoint(self.model, path)
        except TypeError:
            return False
        return True

    def load(self, path: str) -> bool:
        """
        Loads a model from the path provided.

        :param path: Path to saved model
        :type path: str
        :return: Whether load succeeded or not
        :rtype: bool
        """
        try:
            checkpoint = load_checkpoint(self.model, path)
            if 'CLASSES' in checkpoint['meta']:
                self.model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                warnings.warn(
                    'Class names are not saved in the checkpoint\'s meta data, use Cityscapes classes by default.')
                self.model.CLASSES = get_classes('cityscapes')
            self._is_model_trained = True
        except RuntimeError:
            return False
        return True

    def optimize(self, target_device: str) -> bool:
        # Not needed for this learner.
        raise NotImplementedError

    def reset(self) -> None:
        # Not needed for this learner since it is stateless.
        raise NotImplementedError

    @staticmethod
    def download(path: str, mode: str='model', trained_on: str='cityscapes') -> str:
        """
        Download data from the OpenDR server. Valid modes include pre-trained model weights and data used in the unit tests.

        Currently, the following pre-trained models are available:
            - Cityscapes
            - KITTI panoptic segmentation dataset

        :param path: Path to save the model weights
        :type path: str
        :param mode: What kind of data to download
        :type mode: str
        :param trained_on: Dataset on which the model has been trained [applicable only to mode == 'model']
        :type trained_on: str
        :return: Path to the downloaded file
        :rtype: str
        """
        # ToDo: Adjust URLs
        if mode == 'model':
            models = {
                'cityscapes': f'{OPENDR_SERVER_URL}perception/panoptic_segmentation/models/model_cityscapes.pth',
                'kitti': f'{OPENDR_SERVER_URL}perception/panoptic_segmentation/models/model_kitti.pth'
            }
            if trained_on not in models.keys():
                raise ValueError(f'Could not find model weights pre-trained on {trained_on}. '
                                 f'Valid options are {list(models.keys())}')
            url = models[trained_on]
        elif mode == 'test_data':
            url = f'{OPENDR_SERVER_URL}perception/panoptic_segmentation/test_data/test_data.zip'
        else:
            raise ValueError('Invalid mode. Valid options are ["model", "test_data"]')

        filename = os.path.join(path, url.split('/')[-1])

        def pbar_hook(pbar: tqdm):
            prev_b = [0]

            def update_to(b=1, bsize=1, total=None):
                if total is not None:
                    pbar.total = total
                pbar.update((b - prev_b[0]) * bsize)
                prev_b[0] = b

            return update_to

        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f'Downloading {filename}') as pbar:
            urllib.request.urlretrieve(url, filename, pbar_hook(pbar))
        return filename

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

        :return: number of workers
        :rtype: int
        """
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value: int):
        """
        Setter for number of workers used in the data loaders. This will perform the necessary type and value checking.

        :param value: number of workers
        :type value: int
        """
        if not isinstance(value, int):
            raise TypeError('num_workers should be an integer.')
        elif value < 0:
            raise ValueError('num_workers cannot be negative.')
        self._num_workers = value
