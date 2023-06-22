# YOLOv5DetectorLearner Demos

This folder contains minimal code usage examples that showcase the basic inference function of the YOLOv5DetectorLearner 
provided by OpenDR. Specifically the following examples are provided:
1. inference_demo.py: Perform inference on a single image. Setting `--device cpu` performs inference on CPU.
2. webcam_demo.py: A simple tool that performs live object detection using a webcam.
3. inference_tutorial.ipynb: Perform inference using pretrained or custom models.
4. convert_detection_dataset.py: An example of how to convert a `DetectionDataset` into the required format
to train a custom model. Training instructions can be found on the original 
   [YOLOv5 repository](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#3-train)