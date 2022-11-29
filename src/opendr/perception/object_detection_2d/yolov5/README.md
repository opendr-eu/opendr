OpenDR 2D Object Detection - YOLOv5
======

This folder contains the OpenDR Learner class for YOLOv5 for 2D object detection.

Sources
------
The models are taken from the
[Ultralytics implementation](https://github.com/ultralytics/yolov5) with modifications
to make it compatible with OpenDR specifications. Only inference is supported.

Usage
------
- The ```model_name``` parameter is used to specify which model will be loaded. Available models: ```['yolov5s', 'yolov5n', 'yolov5m', 'yolov5l', 'yolov5x', 'yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'custom']```
- For custom models, the ```path``` parameter must be set to point to the location of the weights file.
- The ```temp_path``` folder is used to save the downloaded weights when using pretrained models.
- The ```force_reload``` parameter redownloads the pretrained model when set to `True`. This fixes issues with caching.