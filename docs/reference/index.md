# OpenDR Toolkit Reference Manual

Release 0.1

![OpenDR](images/opendr_logo.png)

Copyright &copy; 2020-2021 OpenDR Project.
OpenDR is funded from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449.

Permission to use, copy and distribute this documentation for any purpose and without fee is hereby granted in perpetuity, provided that no modifications are made to this documentation.

The copyright holder makes no warranty or condition, either expressed or implied, including but not limited to any implied warranties of merchantability and fitness for a particular purpose, regarding this manual and the associated software.
This manual is provided on an `as-is` basis.
Neither the copyright holder nor any applicable licensor will be liable for any incidental or consequential damages.

## Table of Contents

- [Inference and Training API](inference-and-training-api.md)
    - `engine` Module
        - [engine.data Module](engine-data.md)
        - [engine.datasets Module](engine-datasets.md)
        - [engine.target Module](engine-target.md)
    - `perception` Module
        - [face_recognition_learner Module](face-recognition.md)
        - [lightweight-open-pose Module](lightweight-open-pose.md)
        - [activity_recognition Module](activity-recognition.md)
        - [edgespeechnets Module](edgespeechnets.md)
        - [quadraticselfonn Module](quadratic-selfonn.md)
        - [rgbd_hand_gesture_learner Module](rgbd_hand_gesture_learner.md)
        - [gated_recurrent_unit](gated-recurrent-unit-learner.md)
        - [matchboxnet Module](matchboxnet.md)
        - [voxel-object-detection-3d Module](voxel-object-detection-3d.md)
        - [object_tracking_2d_fair_mot Module](object-tracking-2d-fair-mot.md)
        - [object_tracking_3d_ab3dmot Module](object-tracking-3d-ab3dmot.md)
        - [multilinear_compressive_learning Module](multilinear_compressive_learning.md)
        - [skeleton_based_action_recognition](skeleton_based_action_recognition.md)
        - [semantic-segmentation Module](semantic-segmentation.md)
        - [landmark_based_facial_expression_recognition](landmark-based-facial-expression-recognition.md)

    - `control` Module
      - [mobile_manipulation Module](mobile-manipulation.md)
- [ROSBridge Package](rosbridge.md)
- [C Inference API](c-api.md)
    - [data.h](c-data-h.md)
    - [target.h](c-target-h.md) 
    - [opendr_utils.h](c-opendr-utils-h.md)
    - [face_recognition.h](c-face-recognition-h.md)
