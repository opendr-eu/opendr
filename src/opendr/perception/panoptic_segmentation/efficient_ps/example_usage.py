from pprint import pprint
import numpy as np
import mmcv

from torch.utils.data import DataLoader

from opendr.perception.panoptic_segmentation.efficient_ps.src.opendr_interface.efficient_ps_learner import \
    EfficientPsLearner
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset, CityscapesDataset2
# from opendr.perception.panoptic_segmentation.datasets.cityscapes import Image
from opendr.engine.data import Image
from opendr.engine.target import Heatmap


def main():
    dataset = CityscapesDataset(path='/home/voedisch/git/EfficientPS/data/cityscapes')
    dataset2 = CityscapesDataset2(path='/home/voedisch/git/EfficientPS/data/dr_cityscapes/training')
    dataset3 = CityscapesDataset2(path='/home/voedisch/git/EfficientPS/data/dr_cityscapes/test')

    learner = EfficientPsLearner(
        lr=0.07,
        iters=1,
        batch_size=2,
        optimizer='SGD',
        device='cuda:0'
    )

    # learner.save(path='/home/voedisch/git/EfficientPS/checkpoints/test/model.pth')

    # eval_results = learner.eval(dataset, print_results=True)

    # learner.fit(dataset2, val_dataset=dataset3)

    learner.load(path='/home/voedisch/git/EfficientPS/checkpoints/efficientPS_cityscapes/model/model.pth')
    eval_results = learner.eval2(dataset3, print_results=True)
    return

    image_filenames = [
        '/home/voedisch/git/EfficientPS/data/dr_cityscapes/test/images/lindau_000001_000019.png',
        '/home/voedisch/git/EfficientPS/data/dr_cityscapes/test/images/lindau_000002_000019.png',
        '/home/voedisch/git/EfficientPS/data/dr_cityscapes/test/images/lindau_000003_000019.png',
        '/home/voedisch/git/EfficientPS/data/dr_cityscapes/test/images/lindau_000004_000019.png',
        '/home/voedisch/git/EfficientPS/data/dr_cityscapes/test/images/lindau_000006_000019.png',
        '/home/voedisch/git/EfficientPS/data/dr_cityscapes/test/images/lindau_000007_000019.png',
        '/home/voedisch/git/EfficientPS/data/dr_cityscapes/test/images/lindau_000008_000019.png'
    ]
    images = [Image(mmcv.imread(f)) for f in image_filenames]


    predictions = learner.infer(images)

    print('done')


if __name__ == "__main__":
    main()
