OpenDR 2D Object Detection - Nanodet
======

This folder contains the OpenDR Learner class for Nanodet for 2D object detection.

Sources
------
Large parts of the implementation are taken from [Nanodet Github](https://github.com/RangiLyu/nanodet) with modifications to make it compatible with OpenDR specifications.

Usage
------
- For VOC and COCO like datasets, an ```ExternalDataset``` with the root path and dataset name (```voc```, ```coco```) must be passed to the fit function.
- The ```temp_path``` folder is used to save checkpoints during training.