import mmcv

from pathlib import Path
from typing import List, Tuple

from opendr.perception.panoptic_segmentation.efficient_ps.src.opendr_interface.efficient_ps_learner import \
    EfficientPsLearner
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset
from opendr.engine.data import Image
from opendr.engine.target import Heatmap

DATA_ROOT = '/home/voedisch/data'

def train():
    train_dataset = CityscapesDataset(path=f'{DATA_ROOT}/cityscapes/training')
    val_dataset = CityscapesDataset(path=f'{DATA_ROOT}/cityscapes/test')

    learner = EfficientPsLearner(
        iters=2,
        batch_size=1,
        checkpoint_after_iter=2,
        device='cuda:0',
        config_file=str(Path(__file__).parent / 'configs' / 'efficientPS_singlegpu_sample.py')
    )
    learner.fit(train_dataset, val_dataset=val_dataset, logging_path=str(Path(__file__).parent / 'work_dir'),
                silent=True)
    learner.save(path=f'{DATA_ROOT}/checkpoints/sample/model.path')


def evaluate():
    val_dataset = CityscapesDataset(path=f'{DATA_ROOT}/cityscapes/test')

    learner = EfficientPsLearner(
        device='cuda:0',
        config_file=str(Path(__file__).parent / 'configs' / 'efficientPS_singlegpu_sample.py')
    )
    learner.load(path=f'{DATA_ROOT}/checkpoints/efficientPS_cityscapes/model/model.pth')
    learner.eval(val_dataset, print_results=True)


def inference():
    image_filenames = [
        f'{DATA_ROOT}/cityscapes/test/images/lindau_000001_000019.png',
        f'{DATA_ROOT}/cityscapes/test/images/lindau_000002_000019.png',
        f'{DATA_ROOT}/cityscapes/test/images/lindau_000003_000019.png',
    ]
    images = [Image(mmcv.imread(f)) for f in image_filenames]

    learner = EfficientPsLearner(
        device='cuda:0',
        config_file=str(Path(__file__).parent / 'configs' / 'efficientPS_singlegpu_sample.py')
    )
    learner.load(path=f'{DATA_ROOT}/checkpoints/efficientPS_cityscapes/model/model.pth')
    predictions: List[Tuple[Heatmap, Heatmap]] = learner.infer(images)


if __name__ == "__main__":
    train()
    print('-' * 40 + '\n===> Training succeeded\n' + '-' * 40)
    evaluate()
    print('-' * 40 + '\n===> Evaluation succeeded\n' + '-' * 40)
    inference()
    print('-' * 40 + '\n===> Inference succeeded\n' + '-' * 40)
