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

import time
import os
import shutil

import numpy as np
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler

from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint, Runner
from mmcv.parallel import scatter, collate, MMDataParallel

from opendr.engine.learners import Learner
from opendr.engine.target import Heatmap
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset, Image

from mmdet.core import get_classes, build_optimizer, EvalHook, save_panoptic_eval
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import build_dataloader
from mmdet.apis.train import batch_processor
from mmdet.utils import collect_env


class EfficientPsLearner(Learner):
    def __init__(self,
                 lr: float = .07,
                 iters: int = 160,
                 batch_size: int = 1,
                 optimizer: str = 'SGD',
                 momentum: float = .9,
                 weight_decay: float = .0001,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 device: str = "cuda:0",
                 seed: Optional[float] = None,
                 work_dir: str = str(Path(__file__).parents[2] / 'work_dir'),
                 config_file: str = str(Path(__file__).parents[2] / 'configs' / 'efficientPS_singlegpu_sample.py')
                 ):
        super().__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer, device=device)
        if optimizer_config is None:
            optimizer_config = {'grad_clip': {'max_norm': 35, 'norm_type': 2}}

        self._cfg = Config.fromfile(config_file)
        self._cfg.workflow = [('train', 1)]
        self._cfg.model.pretrained = None
        self._cfg.optimizer = {'type': self.optimizer, 'lr': self.lr, 'momentum': momentum,
                               'weight_decay': weight_decay}
        self._cfg.optimizer_config = optimizer_config
        self._cfg.total_epochs = self.iters
        self._cfg.gpus = 1  # Numbers of GPUs to use (only applicable to non-distributed training)
        self._cfg.seed = seed
        self._cfg.work_dir = work_dir

        # Create model
        self.model = build_detector(self._cfg.model, train_cfg=self._cfg.train_cfg, test_cfg=self._cfg.test_cfg)
        self.model.to(self.device)
        self._is_model_trained = False

    def fit(self,
            dataset,
            val_dataset: Optional[CityscapesDataset] = None,
            silent: Optional[bool] = True,
            verbose: Optional[bool] = True,
            num_workers: int = 1
            ):
        """
        This method is used for training the algorithm on a train dataset and
        validating on a val dataset.

        Can be parameterized based on the learner attributes and custom hyperparameters
        added by the implementation and returns stats regarding training and validation.

        :param dataset: Object that holds the training dataset
        :type dataset: Dataset class type
        :param val_dataset: Object that holds the validation dataset
        :type val_dataset: Dataset class type, optional
        :param verbose: if set to True, enables the maximum logging verbosity (depends on the actual algorithm)
        :type verbose: bool, optional
        :param silent: if set to True, disables printing training progress reports to STDOUT
        :type silent: bool, optional
        """

        dataset.pipeline = self._cfg.train_pipeline
        dataloaders = [build_dataloader(
            dataset.get_mmdet_dataset(),
            self.batch_size,
            num_workers,
            self._cfg.gpus,
            dist=False,
            seed=self._cfg.seed
        )]

        # Put model on GPUs
        self.model = MMDataParallel(self.model, device_ids=range(self._cfg.gpus)).cuda()

        optimizer = build_optimizer(self.model, self._cfg.optimizer)

        # Record some important information such as environment info and seed
        env_info_dict = collect_env()
        env_info = '\n'.join([('{}: {}'.format(k, v)) for k, v in env_info_dict.items()])
        meta = {'env_info': env_info, 'seed': self._cfg.seed}

        runner = Runner(
            self.model,
            batch_processor,
            optimizer,
            self._cfg.work_dir,
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
                workers_per_gpu=num_workers,
                dist=False,
                shuffle=False
            )
            runner.register_hook(EvalHook(val_dataloader, interval=1, metric=['panoptic']))

        runner.run(dataloaders, self._cfg.workflow, self.iters)

    def eval(self,
             dataset: Any,
             num_workers: int = 0,
             tmp_directory: str = 'eval_tmp_dir',
             print_results: bool = False
             ):
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=num_workers,
                                collate_fn=lambda x: x)

        with tqdm(dataloader, unit='batch') as pbar:
            for i, batch in enumerate(dataloader):
                images = [data[0] for data in batch]
                predictions = self.infer(images, return_raw_logits=True)
                save_panoptic_eval(predictions, path=tmp_directory)
                pbar.update(1)

        results = dataset.evaluate(os.path.join(tmp_directory, 'tmp'), os.path.join(tmp_directory, 'tmp_json'))

        if print_results:
            msg = f"{'Category':<14s}| {'PQ':>5s} {'SQ':>5s} {'RQ':>5s} {'N':>5s}\n"
            msg += "-" * 41 + "\n"
            for x in ['All', 'Things', 'Stuff']:
                msg += f"{x:<14s}| {results[x]['pq'] * 100:>5.1f} {results[x]['sq'] * 100:>5.1f} {results[x]['rq'] * 100:>5.1f} {results[x]['n']:>5d}\n"
            msg += "-" * 41 + "\n"
            for cat, value in results['per_class'].items():
                msg += f"{cat:<14s}| {value['pq'] * 100:>5.1f} {value['sq'] * 100:>5.1f} {value['rq'] * 100:>5.1f}\n"
            msg = msg[:-1]
            print(msg)

        shutil.rmtree(tmp_directory)
        return results

    def infer(self,
              batch: Union[Image, List[Image]],
              return_raw_logits: bool = False
              ) -> Union[List[Tuple[Heatmap, Heatmap]], np.ndarray]:
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
        if isinstance(batch, Image):
            batch = [Image]
        mmdet_batch = []
        for img in batch:
            try:
                filename = img.filename
            except AttributeError:
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

    def download(self) -> bool:
        raise NotImplementedError

    @property
    def config(self):
        return self._cfg
