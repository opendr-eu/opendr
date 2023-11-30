OpenDR 2D Object Detection - Nanodet
======

This folder contains the OpenDR Learner class for Nanodet for 2D object detection.

Sources
------
Large parts of the implementation are taken from [Nanodet Github](https://github.com/RangiLyu/nanodet) with modifications to make it compatible with OpenDR specifications.

Some enhancements were implemented, drawing inspiration from the [YOLOv5 GitHub](https://github.com/ultralytics/yolov5).
The primary scripts involved are `autobatch.py` and `torch_utils.py`, along with the dataset caching capabilities during training.

Usage
------
- For VOC and COCO like datasets, an ```ExternalDataset``` with the root path and dataset name (```voc```, ```coco```) must be passed to the fit function.
- The ```temp``` folder is used to save checkpoints during training.