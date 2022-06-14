# OpenDR Toolkit Change Log

## Version 1.1
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
