import mmcv

from pathlib import Path
from typing import List, Tuple

from opendr.perception.panoptic_segmentation.efficient_ps.src.opendr_interface.efficient_ps_learner import \
    EfficientPsLearner
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset
from opendr.engine.data import Image
from opendr.engine.target import Heatmap


def train():
    train_dataset = CityscapesDataset(path='/home/voedisch/data/cityscapes/training')
    val_dataset = CityscapesDataset(path='/home/voedisch/data/cityscapes/test')

    learner = EfficientPsLearner(
        iters=2,
        batch_size=1,
        device='cuda:0',
        work_dir=str(Path(__file__).parent / 'work_dir'),
        config_file=str(Path(__file__).parent / 'configs' / 'efficientPS_singlegpu_sample.py')
    )
    learner.fit(train_dataset, val_dataset=val_dataset)
    learner.save(path='/home/voedisch/data/checkpoints/sample')


def evaluate():
    val_dataset = CityscapesDataset(path='/home/voedisch/data/cityscapes/test')

    learner = EfficientPsLearner(
        device='cuda:0',
        config_file=str(Path(__file__).parent / 'configs' / 'efficientPS_singlegpu_sample.py')
    )
    learner.load(path='/home/voedisch/data/checkpoints/efficientPS_cityscapes/model/model.pth')
    learner.eval(val_dataset, print_results=True)


def inference():
    image_filenames = [
        '/home/voedisch/data/cityscapes/test/images/lindau_000001_000019',
        '/home/voedisch/data/cityscapes/test/images/lindau_000002_000019',
        '/home/voedisch/data/cityscapes/test/images/lindau_000003_000019',
    ]
    images = [Image(mmcv.imread(f)) for f in image_filenames]

    learner = EfficientPsLearner(
        device='cuda:0',
        config_file=str(Path(__file__).parent / 'configs' / 'efficientPS_singlegpu_sample.py')
    )
    learner.load(path='/home/voedisch/data/checkpoints/efficientPS_cityscapes/model/model.pth')
    predictions: List[Tuple[Heatmap, Heatmap]] = learner.infer(images)


if __name__ == "__main__":
    train()
    evaluate()
    inference()
