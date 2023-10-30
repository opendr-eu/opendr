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
## for EfficientPS
1. Download the raw dataset from the [EfficientPS website](http://panoptic.cs.uni-freiburg.de/).
2. Extract the file.
3. Convert the files to the expected folder structure and generate panoptic ground truth data for evaluation 
```python
from opendr.perception.panoptic_segmentation import KittiDataset
DOWNLOAD_PATH = '~/data/KITTI-panoptic-segmentation-dataset'
DATA_ROOT = '~/data/kitti'
KittiDataset.prepare_data(DOWNLOAD_PATH, DATA_ROOT)
```

## for EfficientLPS
For using the EfficientLPS module with LiDAR data, the dataset must be downloaded from the original
[source](http://www.semantic-kitti.org/dataset.html). Creating an account may be required.

1. Download the following files:
   - [KITTI Odometry Benchmark Velodyne Point Clouds](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip)
   - [KITTI Odometry Benchmark Calibration Data](http://www.cvlibs.net/download.php?file=data_odometry_calib.zip)
   - [KITTI SemanticKITTI Label Data](http://www.semantic-kitti.org/assets/data_odometry_labels.zip)
   
2. Extract the contents of the `zip` files into the same folder. 
The folder structure should look like the following:  
"unzipped-semantickitti/"   
      &emsp;└── data_odometry_calib/  
      &emsp;└── data_odometry_labels/  
      &emsp;└── data_odometry_velodyne/  

 ```python
from opendr.perception.panoptic_segmentation import SemanticKittiDataset
DOWNLOAD_ROOT = "~/data/unzipped-semantickitti"
DATA_ROOT = "~/data/semantickitti"
SemanticKittiDataset.prepare_data(DOWNLOAD_ROOT, DATA_ROOT)
```