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
from typing import List, Tuple

import cv2

from opendr.engine.data import Image
from opendr.engine.target import Heatmap
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset, KittiDataset
from opendr.perception.panoptic_segmentation.efficient_ps import EfficientPsLearner

DATA_ROOT = '/home/user/data'
CITYSCAPES_ROOT = f'{DATA_ROOT}/cityscapes_pt'
KITTI_ROOT = f'{DATA_ROOT}/kitti_pt'


def download_models():
    EfficientPsLearner.download(f'{DATA_ROOT}/checkpoints/efficientPS/', trained_on='cityscapes')
    EfficientPsLearner.download(f'{DATA_ROOT}/checkpoints/efficientPS/', trained_on='kitti')


def prepare_dataset():
    CityscapesDataset.prepare_data('/home/user/data/cityscapes', CITYSCAPES_ROOT)
    KittiDataset.prepare_data('/home/user/data/kitti_panoptic', KITTI_ROOT)


def train():
    train_dataset = CityscapesDataset(path=f'{CITYSCAPES_ROOT}/training')
    val_dataset = CityscapesDataset(path=f'{CITYSCAPES_ROOT}/test')

    learner = EfficientPsLearner(
        iters=2,
        batch_size=1,
        checkpoint_after_iter=2,
        device='cuda:0',
        config_file=str(Path(__file__).parent / 'configs' / 'singlegpu_sample.py')
    )
    learner.fit(train_dataset, val_dataset=val_dataset, logging_path=str(Path(__file__).parent / 'work_dir'))
    learner.save(path=f'{DATA_ROOT}/checkpoints/efficientPS/sample/model.pth')


def evaluate():
    val_dataset = CityscapesDataset(path=f'{CITYSCAPES_ROOT}/test')

    learner = EfficientPsLearner(
        device='cuda:0',
        config_file=str(Path(__file__).parent / 'configs' / 'singlegpu_sample.py')
    )
    learner.load(path=f'{DATA_ROOT}/checkpoints/efficientPS/kitti/model.pth')
    learner.eval(val_dataset, print_results=True)


def inference():
    image_filenames = [
        f'{CITYSCAPES_ROOT}/val/images/lindau_000001_000019.png',
        f'{CITYSCAPES_ROOT}/val/images/lindau_000002_000019.png',
        f'{CITYSCAPES_ROOT}/val/images/lindau_000003_000019.png',
    ]
    images = [Image(cv2.imread(f)) for f in image_filenames]

    learner = EfficientPsLearner(
        device='cuda:0',
        config_file=str(Path(__file__).parent / 'configs' / 'singlegpu_sample.py')
    )
    learner.load(path=f'{DATA_ROOT}/checkpoints/efficientPS/cityscapes/model.pth')
    predictions: List[Tuple[Heatmap, Heatmap]] = learner.infer(images)


if __name__ == "__main__":
    download_models()
    prepare_dataset()

    train()
    print('-' * 40 + '\n===> Training succeeded\n' + '-' * 40)
    evaluate()
    print('-' * 40 + '\n===> Evaluation succeeded\n' + '-' * 40)
    inference()
    print('-' * 40 + '\n===> Inference succeeded\n' + '-' * 40)
