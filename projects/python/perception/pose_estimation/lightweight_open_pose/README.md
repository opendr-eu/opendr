# Lightweight OpenPose Demos

This folder contains sample applications that demonstrate various parts of the functionality provided by the Lightweight OpenPose tool provided by OpenDR.

More specifically, the following applications are provided:

1. demo/inference_tutorial.ipynb: A step-by-step tutorial on how to run inference using OpenDR's implementation of Pose Estimation
2. demos/benchmarking_demo.py: A simple benchmarking tool for measuring the performance of the tool in various platforms
3. demos/eval_demo.py: A tool that demonstrates how to perform evaluation
4. demos/inference_demo.py: A tool that demonstrates how to perform inference on a single image and then draw the detected poses
5. demos/webcam_demo.py: A simple tool that performs live pose estimation using a webcam
6. demos/wave_detection_demo.py: A rule-based method for wave detection using a webcam
7. jetbot: A demo developed on Webots to demonstrate how OpenDR functionality can be used to provide a naive fall detector. This demo can be also directly used on an NVIDIA JetBot.

Please use the `--device cpu` flag for the demos if you are running them on a machine without a CUDA-enabled GPU. The JetBot demo auto-detects whether a GPU is available.
