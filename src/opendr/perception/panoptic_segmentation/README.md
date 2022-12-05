# Panoptic Segmentation

Panoptic segmentation combines both semantic segmentation and instance segmentation in a single task.
While distinct foreground objects, e.g., cars or pedestrians, receive instance-wise segmentation masks, background classes such as buildings or road surface are combined in class-wide labels. 

## Modules

### EfficientPS: Efficient Panoptic Segmentation

For the task of panoptic segmentation, EfficientPS has been included in the OpenDR toolkit.
The model architecture leverages a shared backbone for efficient encoding and fusing of semantically rich multi-scale features.
Two separate network heads create predictions for semantic and instance segmentation, respectively.
The final panoptic fusion model combines the output of the task-specific heads into a single panoptic segmentation map.

Website: http://panoptic.cs.uni-freiburg.de <br>
Arxiv: https://arxiv.org/abs/2004.02307 <br>
GitHub repository: https://github.com/DeepSceneSeg/EfficientPS

**BibTeX**:
```bibtex
@article{mohan2020efficientps,
  title={EfficientPS: Efficient Panoptic Segmentation},
  author={Mohan, Rohit and Valada, Abhinav},
  journal={International Journal of Computer Vision (IJCV)},
  year={2021}
}
```

**Base repositories**

The OpenDR implementation extends the [EfficientPS repository](https://github.com/DeepSceneSeg/EfficientPS), from [Rohit Mohan](https://rl.uni-freiburg.de/people/mohan) and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), with the OpenDR interface.

Please note that the original repository is heavily based on
- [mmdetection](https://github.com/open-mmlab/mmdetection) by the [OpenMMLab](https://openmmlab.com/) project
- [gen-efficient-net-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch) authored by [Ross Wightman](https://github.com/rwightman)

## Example Usage

More code snippets can be found in [example_usage.py](../../../../projects/python/perception/panoptic_segmentation/efficient_ps/example_usage.py) with the corresponding [readme](../../../../projects/python/perception/panoptic_segmentation/efficient_ps/README.md).

**Prepare the downloaded Cityscapes dataset** (see the [datasets' readme](./datasets/README.md) as well)
```python
from opendr.perception.panoptic_segmentation import CityscapesDataset
DOWNLOAD_PATH = '~/data/cityscapes_raw'
DATA_ROOT = '~/data/cityscapes'
CityscapesDataset.prepare_data(DOWNLOAD_PATH, DATA_ROOT)
```

**Run inference and visualize result**
```python
from opendr.engine.data import Image
from opendr.perception.panoptic_segmentation import EfficientPsLearner
DATA_ROOT = '~/data/cityscapes'
image_filenames = [
    f'{DATA_ROOT}/val/images/lindau_000001_000019.png',
    f'{DATA_ROOT}/val/images/lindau_000002_000019.png',
    f'{DATA_ROOT}/val/images/lindau_000003_000019.png',
]
images = [Image.open(f) for f in image_filenames]
config_file = 'singlegpu_cityscapes.py' # stored in efficient_ps/configs
learner = EfficientPsLearner(config_file)
learner.load('model.pth') # alternatively, one can just specify the path to the folder
predictions = learner.infer(images)
for image, prediction in zip(images, predictions):
    EfficientPsLearner.visualize(image, prediction)
``` 

**Run evaluation**
```python
from opendr.perception.panoptic_segmentation import EfficientPsLearner, CityscapesDataset
DATA_ROOT = '~/data/cityscapes'
val_dataset = CityscapesDataset(path=f'{DATA_ROOT}/val')
config_file = 'singlegpu_cityscapes.py' # stored in efficient_ps/configs
learner = EfficientPsLearner(config_file)
learner.load('model.pth') # alternatively, one can just specify the path to the folder
learner.eval(val_dataset, print_results=True)
```

**Run training**
```python
from opendr.perception.panoptic_segmentation import EfficientPsLearner, CityscapesDataset
DATA_ROOT = '~/data/cityscapes'
train_dataset = CityscapesDataset(path=f'{DATA_ROOT}/training')
val_dataset = CityscapesDataset(path=f'{DATA_ROOT}/val')
config_file = 'singlegpu_cityscapes.py' # stored in efficient_ps/configs
learner = EfficientPsLearner(config_file)
learner.fit(train_dataset, val_dataset)
```
