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

import sys
from pathlib import Path

from opendr.engine.data import Image
from opendr.perception.panoptic_segmentation import EfficientPsLearner, CityscapesDataset, KittiDataset

DATA_ROOT = '/home/USER/data/efficientPS'
CITYSCAPES_ROOT = f'{DATA_ROOT}/converted_datasets/cityscapes'
KITTI_ROOT = f'{DATA_ROOT}/converted_datasets/kitti_panoptic'


def download_models():
    EfficientPsLearner.download(f'{DATA_ROOT}/checkpoints', trained_on='cityscapes')
    EfficientPsLearner.download(f'{DATA_ROOT}/checkpoints', trained_on='kitti')


def prepare_dataset():
    # These methods require downloading the data first as described in the README in the datasets folder
    CityscapesDataset.prepare_data(f'{DATA_ROOT}/raw_datasets/cityscapes', CITYSCAPES_ROOT)
    KittiDataset.prepare_data(f'{DATA_ROOT}/raw_datasets/kitti_panoptic', KITTI_ROOT)


def train():
    train_dataset = CityscapesDataset(path=f'{CITYSCAPES_ROOT}/train')
    val_dataset = CityscapesDataset(path=f'{CITYSCAPES_ROOT}/val')

    config_file = Path(sys.modules[
                           EfficientPsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_cityscapes.py'
    learner = EfficientPsLearner(
        str(config_file),
        iters=2,
        batch_size=1,
        checkpoint_after_iter=2
    )
    train_stats = learner.fit(train_dataset, val_dataset=val_dataset,
                              logging_path=str(Path(__file__).parent / 'work_dir'))
    learner.save(path=f'{DATA_ROOT}/checkpoints/efficientPS')
    assert train_stats  # This assert is just a workaround since pyflakes does not support the NOQA comment


def evaluate():
    val_dataset = CityscapesDataset(path=f'{CITYSCAPES_ROOT}/val')
    config_file = Path(sys.modules[EfficientPsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_cityscapes.py'
    learner = EfficientPsLearner(str(config_file))
    learner.load(path=f'{DATA_ROOT}/checkpoints/model_cityscapes.pth')
    eval_stats = learner.eval(val_dataset, print_results=True)
    assert eval_stats  # This assert is just a workaround since pyflakes does not support the NOQA comment

    val_dataset = KittiDataset(path=f'{KITTI_ROOT}/val')
    config_file = Path(sys.modules[EfficientPsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_kitti.py'
    learner = EfficientPsLearner(str(config_file))
    learner.load(path=f'{DATA_ROOT}/checkpoints/model_kitti.pth')
    eval_stats = learner.eval(val_dataset, print_results=True)
    assert eval_stats  # This assert is just a workaround since pyflakes does not support the NOQA comment


def inference():
    image_filenames = [
        f'{CITYSCAPES_ROOT}/val/images/lindau_000001_000019.png',
        f'{CITYSCAPES_ROOT}/val/images/lindau_000002_000019.png',
        f'{CITYSCAPES_ROOT}/val/images/lindau_000003_000019.png',
    ]
    images = [Image.open(f) for f in image_filenames]

    config_file = Path(sys.modules[EfficientPsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_cityscapes.py'
    learner = EfficientPsLearner(str(config_file))
    learner.load(path=f'{DATA_ROOT}/checkpoints/model_cityscapes.pth')
    predictions = learner.infer(images)
    for image, prediction in zip(images, predictions):
        EfficientPsLearner.visualize(image, prediction)


if __name__ == "__main__":
    download_models()
    prepare_dataset()

    train()
    print('-' * 40 + '\n===> Training succeeded\n' + '-' * 40)
    evaluate()
    print('-' * 40 + '\n===> Evaluation succeeded\n' + '-' * 40)
    inference()
    print('-' * 40 + '\n===> Inference succeeded\n' + '-' * 40)
