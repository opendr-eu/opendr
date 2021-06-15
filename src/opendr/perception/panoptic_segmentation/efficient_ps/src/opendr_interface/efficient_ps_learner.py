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
import os
import shutil

import numpy as np
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint
from mmcv.parallel import scatter, collate

from opendr.engine.learners import Learner
# from opendr.engine.data import Image
from opendr.engine.target import Heatmap
from opendr.perception.panoptic_segmentation.efficient_ps.src.opendr_interface.util import collate_fn, \
    prepare_results_for_evaluation
from opendr.perception.panoptic_segmentation.datasets.cityscapes import Image

from mmdet.core import get_classes, save_panoptic_eval
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose

from cityscapesscripts.evaluation.evalPanopticSemanticLabeling import pq_compute_multi_core, average_pq, pq_compute_single_core


class EfficientPsLearner(Learner):
    def __init__(self,
                 lr: float,
                 iters: int,
                 batch_size: int = 1,
                 optimizer: str = 'sgd',
                 device: str = "cuda:0"
                 ):
        super().__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer, device=device)

        # ToDo: figure out configuration
        CONFIG_FILE = str(Path(__file__).parents[2] / 'configs' / 'efficientPS_singlegpu_sample.py')
        self._cfg = Config.fromfile(CONFIG_FILE)
        self._cfg.model.pretrained = None

        # Create model
        self.model = build_detector(self._cfg.model, train_cfg=self._cfg.train_cfg, test_cfg=self._cfg.test_cfg)
        self.model.to(self.device)
        self._is_model_trained = False
        # self.model.cfg = self._cfg # save the config in the model for convenience

        # self._dataset = build_dataset(self._cfg.data.train)

    def fit(self,
            dataset,
            val_dataset: Any = None,
            logging_path: Optional[str] = None,
            silent: Optional[bool] = True,
            verbose: Optional[bool] = True
            ) -> Dict[str, Any]:
        # ToDo: missing additional parameters
        # train_detector(self._model, self._dataset, self._cfg)

        raise NotImplementedError

    def eval(self,
             dataset: Any,
             tmp_folder: str = os.path.join(os.getcwd(), 'tmp'),
             num_workers: int = 0,
             print_results: bool = False,
             ) -> Dict[str, Any]:
        """
        This method is used to evaluate the algorithm on a dataset
        and returns stats regarding the evaluation ran.

        The following stats are computed:
        - Panoptic quality (pq)
        - Sementation quality (sq)
        - Recognition quality (rq)

        :param dataset: Object that holds the dataset to evaluate the algorithm on
        :type dataset: Dataset class type
        :param tmp_folder: Path to create a temporary folder to save the predictions
        :type tmp_folder: str
        :param num_workers: Number of workers to be used for loading the dataset
        :type num_workers: int
        :param print_results: If set to true, the results will be printed in a table to STDOUT
        :type print_results: bool
        :return: Returns stats regarding evaluation
        :rtype: dict
        """
        if self.model is None:
            raise RuntimeError('No model loaded.')
        if not self._is_model_trained:
            warnings.warn('The current model has not been trained.')

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=num_workers,
                                collate_fn=collate_fn)

        matched_annotations_list = []
        with tqdm(dataloader, unit='batch') as pbar:
            for i, batch in enumerate(dataloader):
                images = [data[0] for data in batch]
                results = self.infer(images, return_raw_logits=True)
                pred_annotation = prepare_results_for_evaluation(results, tmp_folder)
                for gt, pred in zip([data[1] for data in batch], pred_annotation):
                    matched_annotations_list.append((gt, pred))
                pbar.update(1)

        pred_folder = tmp_folder
        if num_workers > 0:
            pq_stat = pq_compute_multi_core(matched_annotations_list, dataset.ground_truth_folder, pred_folder,
                                            dataset.categories)
        else:
            pq_stat = pq_compute_single_core(0, matched_annotations_list, dataset.ground_truth_folder, pred_folder,
                                             dataset.categories)
        eval_results = average_pq(pq_stat, dataset.categories)

        category_ids = sorted(eval_results['per_class'].keys())
        for category_id in category_ids:
            eval_results['per_class'][dataset.categories[category_id]['name']] = eval_results['per_class'].pop(
                category_id)

        if print_results:
            msg = f"{'Category':<14s}| {'PQ':>5s} {'SQ':>5s} {'RQ':>5s} {'N':>5s}\n"
            msg += "-" * 41 + "\n"
            for x in ['All', 'Things', 'Stuff']:
                msg += f"{x:<14s}| {eval_results[x]['pq'] * 100:>5.1f} {eval_results[x]['sq'] * 100:>5.1f} {eval_results[x]['rq'] * 100:>5.1f} {eval_results[x]['n']:>5d}\n"
            msg += "-" * 41 + "\n"
            for cat, value in eval_results['per_class'].items():
                msg += f"{cat:<14s}| {value['pq'] * 100:>5.1f} {value['sq'] * 100:>5.1f} {value['rq'] * 100:>5.1f}\n"
            msg = msg[:-1]
            print(msg)

        shutil.rmtree(tmp_folder)
        return eval_results

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
        test_pipeline = Compose(self._cfg.data.test.pipeline[1:])
        device = next(self.model.parameters()).device

        # Convert to the format expected by the mmdetection API
        if isinstance(batch, Image):
            batch = [Image]
        mmdet_batch = []
        for img in batch:
            mmdet_img = {'filename': img.filename, 'img': img.numpy(), 'img_shape': img.numpy().shape,
                         'ori_shape': img.numpy().shape}
            mmdet_img = test_pipeline(mmdet_img)
            mmdet_batch.append(scatter(collate([mmdet_img], samples_per_gpu=1), [device])[0])

        # Forward the model
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

                    # Some pixels have not gotten a semantic class assigned because they are marked as stuff by the instance head
                    # but not by the semantic segmentation head
                    # We mask them as 255 in the semantic segmentation map
                    # semantic_pred[semantic_pred == 255] = -1

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
            # self.model.to(self.device)
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
