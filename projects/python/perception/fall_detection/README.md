# Fall Detector Demos

This folder contains sample applications that demonstrate various parts of the functionality provided by the Fall Detector algorithm provided by OpenDR.

Specifically, the following applications are provided:

1. demos/eval_demo.py: A tool that demonstrates how to perform evaluation of the Fall Detector algorithm
2. demos/inference_demo.py: A tool that demonstrates how to perform inference on images and then draw the detected poses
3. demos/webcam_demo.py: A simple tool that performs live fall detection using a webcam
4. demos/inference_tutorial.ipynb: A step-by-step tutorial on how to run inference using OpenDR's implementation of rule-based Fall Detection

Please use the `--device cpu` flag for the demos if you are running them on a machine without a CUDA-enabled GPU.
