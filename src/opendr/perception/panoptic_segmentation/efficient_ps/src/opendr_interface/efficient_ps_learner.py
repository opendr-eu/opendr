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

import numpy as np
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint
from mmcv.parallel import scatter, collate

from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import Heatmap

from mmdet.core import get_classes
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose


class EfficientPsLearner(Learner):
    def __init__(self, lr: float, iters: int, optimizer: str, device: str):
        super().__init__(lr=lr, iters=iters, optimizer=optimizer, device=device)

        # ToDo: figure out configuration
        CONFIG_FILE = str(Path(__file__).parents[2] / 'configs' / 'efficientPS_singlegpu_sample.py')
        self._cfg = Config.fromfile(CONFIG_FILE)
        self._cfg.model.pretrained = None

        # Create model
        self.model = build_detector(self._cfg.model, train_cfg=self._cfg.train_cfg, test_cfg=self._cfg.test_cfg)
        # self.model.cfg = self._cfg # save the config in the model for convenience

        # self._dataset = build_dataset(self._cfg.data.train)

    def fit(self,
            dataset,
            val_dataset: Any = None,
            logging_path: Optional[str] = None,
            silent: Optional[bool] = True,
            verbose: Optional[bool] = True) -> Dict[str, Any]:
        # ToDo: missing additional parameters
        # train_detector(self._model, self._dataset, self._cfg)

        raise NotImplementedError

    def eval(self, dataset: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def infer(self, batch: Union[Image, List[Image]]) -> List[Tuple[Heatmap, Heatmap]]:
        """
        This method performs inference on the batch provided.

        :param batch: Object that holds a batch of data to run inference on
        :type batch: Image class type or list of Image class type
        :return: A list of predicted targets
        :rtype: list of tuples of Heatmap class type
        """
        if self.model is None:
            raise RuntimeError('No model loaded.')
        self.model.eval()

        # Build the data pipeline
        test_pipeline = Compose(self._cfg.data.test.pipeline[1:])
        device = next(self.model.parameters()).device

        # Convert to the format expected by the mmdetection API
        if isinstance(batch, Image):
            batch = [Image]
        mmdet_batch = []
        for img in batch:
            mmdet_img = {'filename': None, 'img': img.numpy(), 'img_shape': img.numpy().shape, 'ori_shape': img.numpy().shape}
            mmdet_img = test_pipeline(mmdet_img)
            mmdet_batch.append(scatter(collate([mmdet_img], samples_per_gpu=1), [device])[0])

        # Forward the model
        results = []
        with torch.no_grad():
            for data in mmdet_batch:
                data['eval'] = 'panoptic'
                prediction = self.model(return_loss=False, rescale=True, **data)
                panoptic_pred, cat_pred, _ = prediction[0]
                panoptic_pred = panoptic_pred.numpy()
                semantic_pred = cat_pred[panoptic_pred].numpy()

                # Some pixels have not gotten a semantic class assigned because they are marked as stuff by the instance head
                # but not by the semantic segmentation head
                # We mask them as -1 in the semantic segmentation map
                semantic_pred[semantic_pred == 255] = -1

                panoptic_pred = Heatmap(panoptic_pred.astype(np.uint8), description='Instance prediction')
                semantic_pred = Heatmap(semantic_pred.astype(np.uint8), description='Semantic prediction')
                results.append((panoptic_pred, semantic_pred))

        return results

    def save(self, path: str) -> bool:
        """
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str
        :return: Whether save succeeded or not
        :rtype: bool
        """
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
                warnings.warn('Class names are not saved in the checkpoint\'s meta data, use Cityscapes classes by default.')
                self.model.CLASSES = get_classes('cityscapes')
            self.model.to(self.device)
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
