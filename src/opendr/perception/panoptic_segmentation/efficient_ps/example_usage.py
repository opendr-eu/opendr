from pprint import pprint
import numpy as np
import mmcv

from torch.utils.data import DataLoader

from opendr.perception.panoptic_segmentation.efficient_ps.src.opendr_interface.efficient_ps_learner import \
    EfficientPsLearner
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset
from opendr.engine.data import Image
from opendr.engine.target import Heatmap


def main():
    dataset = CityscapesDataset(path='/home/voedisch/git/EfficientPS/data/cityscapes')

    learner = EfficientPsLearner(
        lr=0.07,
        iters=160,
        batch_size=2,
        optimizer='sgd',
        device='cuda:0'
    )
    learner.load(path='/home/voedisch/git/EfficientPS/checkpoints/efficientPS_cityscapes/model/model.pth')
    # learner.save(path='/home/voedisch/git/EfficientPS/checkpoints/test/model.pth')

    eval_results = learner.eval(dataset, print_results=True)
    return

    image_filenames = [
        # '/home/voedisch/data/cityscapes/leftImg8bit/val/lindau/lindau_000000_000019_leftImg8bit.png',
        '/home/voedisch/data/cityscapes/leftImg8bit/val/lindau/lindau_000001_000019_leftImg8bit.png',
        '/home/voedisch/data/cityscapes/leftImg8bit/val/lindau/lindau_000002_000019_leftImg8bit.png'
    ]
    images = [Image(mmcv.imread(f)) for f in image_filenames]

    predictions = learner.infer(images)

    print('done')


if __name__ == "__main__":
    main()
