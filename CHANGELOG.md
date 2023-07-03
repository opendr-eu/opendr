# OpenDR Toolkit Change Log

## Version 2.2.0
Released on July, 3rd, 2023.

  - Dependency Updates ([#431](https://github.com/opendr-eu/opendr/pull/431)):
    - Updated PyTorch version from 1.9.0 to 1.13.1.
    - Updated Torchvision version from 0.10.0 to 0.14.1.
    - Updated Torchaudio version from 0.9.0 to 0.13.1.
    - Downgraded wheel version to 0.38.4 due to bug in recent version.

## Version 2.1.0
Released on February, 22nd, 2023.

  - New Features:
    - Added Efficient LiDAR Panoptic Segmentation ([#359](https://github.com/opendr-eu/opendr/pull/359)).
    - Added Nanodet 2D Object Detection tool ([#352](https://github.com/opendr-eu/opendr/pull/352)).
    - Added C API implementations of NanoDet 2D Object Detection tool ([#352](https://github.com/opendr-eu/opendr/pull/352)).
    - Added C API implementations of forward pass of DETR 2D Object Detection tool ([#383](https://github.com/opendr-eu/opendr/pull/383)).
    - Added C API implementations of forward pass of DeepSORT 2D Object Tracking tool ([#383](https://github.com/opendr-eu/opendr/pull/383)).
    - Added C API implementations of forward pass of Lightweight OpenPose, Pose Estimator tool ([#383](https://github.com/opendr-eu/opendr/pull/383)).
    - Added C API implementations of forward pass of X3D 2D Activity Recognition tool ([#383](https://github.com/opendr-eu/opendr/pull/383)).
    - Added C API implementations of forward pass of Progressive Spatiotemporal GCN Skeleton-based Action Recognition tool ([#383](https://github.com/opendr-eu/opendr/pull/383)).
    - Added Binary High Resolution Analysis tool ([#402](https://github.com/opendr-eu/opendr/pull/402)).
    - Added Multi-Object-Search tool ([#363](https://github.com/opendr-eu/opendr/pull/363))
  - Enhancements:
    - Added support in C API for detection target structure and vector of detections ([#352](https://github.com/opendr-eu/opendr/pull/352))
    - Added support in C API for tensor structure and vector of tensors ([#383](https://github.com/opendr-eu/opendr/pull/383))
    - Added support in C API for json parser ([#383](https://github.com/opendr-eu/opendr/pull/383))

## Version 2.0.0
Released on December, 31st, 2022.

  - New Features:
    - Added YOLOv5 as an inference-only tool ([#360](https://github.com/opendr-eu/opendr/pull/360)).
    - Added Continual Transformer Encoders ([#317](https://github.com/opendr-eu/opendr/pull/317)).
    - Added Continual Spatio-Temporal Graph Convolutional Networks tool ([#370](https://github.com/opendr-eu/opendr/pull/370)).
    - Added AmbiguityMeasure utility tool ([#361](https://github.com/opendr-eu/opendr/pull/361)).
    - Added SiamRPN 2D tracking tool ([#367](https://github.com/opendr-eu/opendr/pull/367)).
    - Added Facial Emotion Estimation tool ([#264](https://github.com/opendr-eu/opendr/pull/264)).
    - Added High resolution pose estimation tool ([#356](https://github.com/opendr-eu/opendr/pull/356)).
    - Added ROS2 nodes for all included tools ([#256](https://github.com/opendr-eu/opendr/pull/256)).
    - Added missing ROS nodes and homogenized the interface across the tools ([#305](https://github.com/opendr-eu/opendr/issues/305)).
  - Bug Fixes:
    - Fixed `BoundingBoxList`, `TrackingAnnotationList`, `BoundingBoxList3D` and `TrackingAnnotationList3D` confidence warnings ([#365](https://github.com/opendr-eu/opendr/pull/365)).
    - Fixed undefined `image_id` and `segmentation` for COCO `BoundingBoxList` ([#365](https://github.com/opendr-eu/opendr/pull/365)).
    - Fixed Continual X3D ONNX support ([#372](https://github.com/opendr-eu/opendr/pull/372)).
    - Fixed several issues with ROS nodes and improved performance ([#305](https://github.com/opendr-eu/opendr/issues/305)).

## Version 1.1.1
Released on June, 30th, 2022.

  - Bug Fixes:
    - Fix Efficient Panoptic Segmentation submodule commit ([#268](https://github.com/opendr-eu/opendr/pull/268)).
    - Fix Face Recognition compilation error ([#267](https://github.com/opendr-eu/opendr/pull/267)).

## Version 1.1.0
Released on June, 14th, 2022.

  - New Features:
    - Added end-to-end planning tool ([#223](https://github.com/opendr-eu/opendr/pull/223)).
    - Added seq2seq-nms module, along with other custom NMS implementations for 2D object detection.([#232](https://github.com/opendr-eu/opendr/pull/232)).
  - Enhancements:
    - Added support for modular pip packages allowing tools to be installed separately ([#201](https://github.com/opendr-eu/opendr/pull/201)).
    - Simplified the installation process for pip by including the appropriate post-installation scripts ([#201](https://github.com/opendr-eu/opendr/pull/201)).
    - Improved the structure of the toolkit by moving `io` from `utils` to `engine.helper` ([#201](https://github.com/opendr-eu/opendr/pull/201)).
    - Added support for `post-install` scripts and `opendr` dependencies in `.ini` files  ([#201](https://github.com/opendr-eu/opendr/pull/201)).
    - Updated toolkit to support CUDA 11.2 and improved GPU support ([#215](https://github.com/opendr-eu/opendr/pull/215)).
    - Added a standalone pose-based fall detection tool ([#237](https://github.com/opendr-eu/opendr/pull/237))
  - Bug Fixes:
    - Updated wheel building pipeline to include missing files and removed unnecessary dependencies ([#200](https://github.com/opendr-eu/opendr/pull/200)).
    - `panoptic_segmentation/efficient_ps`: updated dataset preparation scripts to create correct validation ground truth ([#221](https://github.com/opendr-eu/opendr/pull/221)).
    - `panoptic_segmentation/efficient_ps`: added specific configuration files for the provided pretrained models ([#221](https://github.com/opendr-eu/opendr/pull/221)).
    - `c_api/face_recognition`: pass key by const reference in `json_get_key_string()` ([#221](https://github.com/opendr-eu/opendr/pull/221)).
    - `pose_estimation/lightweight_open_pose`: fixed height check on transformations.py according to original tool repo ([#242](https://github.com/opendr-eu/opendr/pull/242)).
    - `pose_estimation/lightweight_open_pose`: fixed two bugs where ONNX optimization failed on specific learner parameterization ([#242](https://github.com/opendr-eu/opendr/pull/242)).
  - Dependency Updates:
    - `heart anomaly detection`: upgraded scikit-learn runtime dependency from 0.21.3 to 0.22 ([#198](https://github.com/opendr-eu/opendr/pull/198)).
    - Relaxed all dependencies to allow future versions of non-critical tools to be used ([#201](https://github.com/opendr-eu/opendr/pull/201)).


## Version 1.0
Released on December 31st, 2021.
