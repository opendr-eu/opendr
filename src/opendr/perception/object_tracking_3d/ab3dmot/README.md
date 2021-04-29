# OpenDR 3D Object Tracking - AB3DMOT

This folder contains the OpenDR Learner class implemented for the 3D Object Tracking task. This method uses 3D Kalman filter and a Hungarian algorithm to associate 3D bounding boxes given by a 3D Object Detection method. 

## Sources

The implementation is based on the [AB3DMOT](https://arxiv.org/abs/2008.08063) method.
The tracking evaluation functions are based on the [KITTI evaluation development kit](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) with exception of using 3D IoU for box matching.