OpenDR 2D Object Detection - SSD
======

This folder contains the OpenDR Learner class for Single Shot Detector for 2D object detection.

Sources
------
Large parts of the Learner code are taken from [GluonCV's SSD implementation](https://www.github.com/dmlc/gluon-cv/tree/master/scripts/detection/ssd) with modifications to make it compatible with OpenDR specifications.

Usage
------
- For VOC and COCO, an ```ExternalDataset``` with the root path and dataset name (```voc```, ```coco```) must be passed to the fit function. For custom datasets, see ```DetectionDataset``` and ```WiderPersonDataset```. 
- The ```temp_path``` folder is used to save checkpoints during training.