# Cityscapes

1. Download the raw files (requires creating an account):
    1. Download [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3).
    2. Download [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1).
2. Extract both files.
3. Convert the files to the expected folder structure and generate panoptic ground truth data for evaluation
```python
from opendr.perception.panoptic_segmentation import CityscapesDataset
DOWNLOAD_PATH = '~/data/cityscapes_raw'
DATA_ROOT = '~/data/cityscapes'
CityscapesDataset.prepare_data(DOWNLOAD_PATH, DATA_ROOT)
```

# KITTI Panoptic Segmentation Dataset

1. Download the raw dataset from the [EfficientPS website](http://panoptic.cs.uni-freiburg.de/).
2. Extract the file.
3. Convert the files to the expected folder structure and generate panoptic ground truth data for evaluation 
```python
from opendr.perception.panoptic_segmentation import KittiDataset
DOWNLOAD_PATH = '~/data/KITTI-panoptic-segmentation-dataset'
DATA_ROOT = '~/data/kitti'
KittiDataset.prepare_data(DOWNLOAD_PATH, DATA_ROOT)
```
