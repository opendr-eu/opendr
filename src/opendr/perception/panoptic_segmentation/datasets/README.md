# Cityscapes

1. Download the raw files
    1. Download [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
    2. Download [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
2. Convert the files to the expected folder structure and generate panoptic ground truth data for evaluation
```python
from opendr.perception.panoptic_segmentation.datasets import CityscapesDataset
DOWNLOAD_PATH = '~/data/cityscapes_raw'
DATA_ROOT = '~/data/cityscapes'
CityscapesDataset.prepare_data(DOWNLOAD_PATH, DATA_ROOT)
```
