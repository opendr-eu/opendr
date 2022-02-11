# OpenDR Toolkit Change Log

## Version 1.X
Released on XX, XXth, 2022.

  - New Features:
    - None.
  - Enhancements:
    - None.
  - Bug Fixes:
    - Updated wheel building pipeline to include missing files and removed unnecessary dependencies ([#200](https://github.com/opendr-eu/opendr/pull/200)).
    - `panoptic_segmentation/efficient_ps`: updated dataset preparation scripts to create correct validation ground truth ([#221](https://github.com/opendr-eu/opendr/pull/221)).
    - `panoptic_segmentation/efficient_ps`: added specific configuration files for the provided pretrained models ([#221](https://github.com/opendr-eu/opendr/pull/221)).
    - `c_api/face_recognition`: pass key by const reference in `json_get_key_string()` ([#221](https://github.com/opendr-eu/opendr/pull/221)).
  - Dependency Updates:
    - `heart anomaly detection`: upgraded scikit-learn runtime dependency from 0.21.3 to 0.22 ([#198](https://github.com/opendr-eu/opendr/pull/198)).


## Version 1.0
Released on December 31th, 2021.