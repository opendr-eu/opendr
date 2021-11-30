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
        - `face recognition`:
            - [face_recognition_learner Module](face-recognition.md)
        - `facial expression recognition`:
            - [landmark_based_facial_expression_recognition](landmark-based-facial-expression-recognition.md)
        - `pose estimation`:
            - [lightweight-open-pose Module](lightweight-open-pose.md)
        - `activity recognition`:
            - [activity_recognition Module](activity-recognition.md)
        - `action recognition`:
            - [skeleton_based_action_recognition](skeleton_based_action_recognition.md)
        - `speech recognition`:
            - [matchboxnet Module](matchboxnet.md)
            - [edgespeechnets Module](edgespeechnets.md)
            - [quadraticselfonn Module](quadratic-selfonn.md)
        - `object detection 2d`:
            - [detr Module](detr.md)
            - [gem Module](gem.md)
            - [retinaface Module](face-detection-2d-retinaface.md)
            - [centernet Module](object-detection-2d-ssd.md)
            - [ssd Module](object-detection-2d-ssd.md)
            - [yolov3 Module](object-detection-2d-yolov3.md)
        - `object detection 3d`:
            - [voxel Module](voxel-object-detection-3d.md)
        - `object tracking 2d`:
            - [fair_mot Module](object-tracking-2d-fair-mot.md)
            - [deep_sort Module](object-tracking-2d-deep-sort.md)
        - `object tracking 3d`:
            - [ab3dmot Module](object-tracking-3d-ab3dmot.md)
        - `multimodal human centric`:
            - [rgbd_hand_gesture_learner Module](rgbd_hand_gesture_learner.md)
        - `compressive learning`:
            - [multilinear_compressive_learning Module](multilinear_compressive_learning.md)
        - `semantic segmentation`:
            - [semantic-segmentation Module](semantic-segmentation.md)
        - `panoptic segmentation`:
            - [efficient_ps Module](efficient_ps.md)
        - `heart anomaly detection`:
            - [gated_recurrent_unit Module](gated-recurrent-unit-learner.md)
            - [attention_neural_bag_of_feature_learner Module](attention_neural_bag_of_feature_learner.md)

    - `control` Module
        - [mobile_manipulation Module](mobile-manipulation.md)
    - `simulation` Module
        - [human_model_generation Module](human_model_generation.md)
- [ROSBridge Package](rosbridge.md)
- [C Inference API](c-api.md)
    - [data.h](c-data-h.md)
    - [target.h](c-target-h.md)
    - [opendr_utils.h](c-opendr-utils-h.md)
    - [face_recognition.h](c-face-recognition-h.md)
