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

import os
import sys
from pathlib import Path

from opendr.perception.panoptic_segmentation import EfficientLpsLearner, SemanticKittiDataset

# User must download the datasets and set the paths accordingly
DATA_ROOT = '/home/USER/data/efficientLPS'
SEMANTIC_KITTI_ROOT = f'{DATA_ROOT}/converted_datasets/semantickitti_panoptic'


def download_models():
    EfficientLpsLearner.download(f'{DATA_ROOT}/checkpoints', trained_on='semantickitti')


def train():
    train_dataset = SemanticKittiDataset(path=os.path.join(SEMANTIC_KITTI_ROOT, "eval_data"), split="train")
    val_dataset = SemanticKittiDataset(path=os.path.join(SEMANTIC_KITTI_ROOT, "eval_data"), split="valid")

    config_file = Path(sys.modules[
                           EfficientLpsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_semantickitti.py'
    learner = EfficientLpsLearner(
        str(config_file),
        iters=2,
        batch_size=1,
        checkpoint_after_iter=2
    )
    train_stats = learner.fit(train_dataset, val_dataset=val_dataset,
                              logging_path=str(Path(__file__).parent / 'work_dir'))
    learner.save(path=f'{DATA_ROOT}/checkpoints/efficientLPS')
    assert train_stats  # This assert is just a workaround since pyflakes does not support the NOQA comment


def evaluate():
    val_dataset = SemanticKittiDataset(path=os.path.join(SEMANTIC_KITTI_ROOT, "eval_data"), split="valid")
    config_file = Path(sys.modules[EfficientLpsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_semantickitti.py'
    learner = EfficientLpsLearner(str(config_file))
    learner.load(path=f'{DATA_ROOT}/checkpoints/model_semantickitti.pth')
    eval_stats = learner.eval(val_dataset, print_results=True)
    assert eval_stats  # This assert is just a workaround since pyflakes does not support the NOQA comment


def inference():
    # Pointcloud files
    pcl_filenames = [
            os.path.join(SEMANTIC_KITTI_ROOT, "infer_data", "seq08_f000000.bin"),
            os.path.join(SEMANTIC_KITTI_ROOT, "infer_data", "seq08_f000010.bin"),
        ]
    point_clouds = [SemanticKittiDataset.load_point_cloud(f) for f in pcl_filenames]

    config_file = Path(sys.modules[EfficientLpsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_semantickitti.py'
    learner = EfficientLpsLearner(str(config_file))
    learner.load(path=f'{DATA_ROOT}/checkpoints/model_semantickitti.pth')
    predictions = learner.infer(point_clouds)
    for point_cloud, prediction in zip(point_clouds, predictions):
        EfficientLpsLearner.visualize(point_cloud, (prediction[0], prediction[1]))


if __name__ == "__main__":
    download_models()

    train()
    print('-' * 40 + '\n===> Training succeeded\n' + '-' * 40)
    evaluate()
    print('-' * 40 + '\n===> Evaluation succeeded\n' + '-' * 40)
    inference()
    print('-' * 40 + '\n===> Inference succeeded\n' + '-' * 40)
