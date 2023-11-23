# High Resolution Pose Estimation

This folder contains sample applications that demonstrate various parts of the functionality provided by the High Resolution Pose Estimation algorithm provided by OpenDR.

More specifically, the applications provided are:

1. demos/inference_demo.py: A tool that demonstrates how to perform inference on a single high resolution image and then draw the detected poses. 
2. demos/eval_demo.py: A tool that demonstrates how to perform evaluation using the High Resolution Pose Estimation algorithm on 720p, 1080p and 1440p datasets. 
3. demos/benchmarking_demo.py: A simple benchmarking tool for measuring the performance of High Resolution Pose Estimation in various platforms.
4. demos/webcam_demo.py: A tool that performs live pose estimation with high resolution pose estimation method using a webcam.
    If `--run-comparison` is enabled then it shows the differences between Lightweight OpenPose, and both adaptive and primary methods in High_resolution pose estimation. 

NOTE: All demos can run either with "primary ROI selection" mode or with "adaptive ROI selection".
Use `--method primary` or `--method adaptive` for each case.

