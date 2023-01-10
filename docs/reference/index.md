# OpenDR Toolkit Reference Manual

*Release 2.0.0*
<div align="center">
  <img src="images/opendr_logo.png" />
</div>

Copyright &copy; 2020-2023 OpenDR Project.
OpenDR is funded from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449.

Permission to use, copy and distribute this documentation for any purpose and without fee is hereby granted in perpetuity, provided that no modifications are made to this documentation.

The copyright holder makes no warranty or condition, either expressed or implied, including but not limited to any implied warranties of merchantability and fitness for a particular purpose, regarding this manual and the associated software.
This manual is provided on an `as-is` basis.
Neither the copyright holder nor any applicable licensor will be liable for any incidental or consequential damages.

## Table of Contents

- [Installation](/docs/reference/installation.md)
- [Customization](/docs/reference/customize.md)
- Inference and Training API
    - `engine` Module
        - [engine.data Module](engine-data.md)
        - [engine.datasets Module](engine-datasets.md)
        - [engine.target Module](engine-target.md)
    - `perception` Module
        - face recognition:
            - [face_recognition_learner Module](face-recognition.md)
        - facial expression recognition:
            - [landmark_based_facial_expression_recognition](landmark-based-facial-expression-recognition.md)
            - [image_based_facial_emotion_estimation](image_based_facial_emotion_estimation.md)
        - pose estimation:
            - [lightweight_open_pose Module](lightweight-open-pose.md)
            - [high_resolution_pose_estimation Module](high-resolution-pose-estimation.md)
        - activity recognition:
            - [skeleton-based action recognition](skeleton-based-action-recognition.md)
            - [continual skeleton-based action recognition Module](skeleton-based-action-recognition.md#class-costgcnlearner)
            - [x3d Module](activity-recognition.md#class-x3dlearner)
            - [continual x3d Module](activity-recognition.md#class-cox3dlearner)
            - [continual transformer encoder Module](continual-transformer-encoder.md)
        - speech recognition:
            - [matchboxnet Module](matchboxnet.md)
            - [edgespeechnets Module](edgespeechnets.md)
            - [quadraticselfonn Module](quadratic-selfonn.md)
        - object detection 2d:
            - [nanodet Module](nanodet.md)
            - [detr Module](detr.md)
            - [gem Module](gem.md)
            - [retinaface Module](face-detection-2d-retinaface.md)
            - [centernet Module](object-detection-2d-centernet.md)
            - [ssd Module](object-detection-2d-ssd.md)
            - [yolov3 Module](object-detection-2d-yolov3.md)
            - [yolov5 Module](object-detection-2d-yolov5.md)
            - [seq2seq-nms Module](object-detection-2d-nms-seq2seq_nms.md)
        - object detection 3d:
            - [voxel Module](voxel-object-detection-3d.md)
        - object tracking 2d:
            - [fair_mot Module](object-tracking-2d-fair-mot.md)
            - [deep_sort Module](object-tracking-2d-deep-sort.md)
            - [siamrpn Module](object-tracking-2d-siamrpn.md)
        - object tracking 3d:
            - [ab3dmot Module](object-tracking-3d-ab3dmot.md)
        - multimodal human centric:
            - [rgbd_hand_gesture_learner Module](rgbd-hand-gesture-learner.md)
            - [audiovisual_emotion_recognition_learner Module](audiovisual-emotion-recognition-learner.md)
        - compressive learning:
            - [multilinear_compressive_learning Module](multilinear-compressive-learning.md)
        - semantic segmentation:
            - [semantic_segmentation Module](semantic-segmentation.md)
        - panoptic segmentation:
            - [efficient_ps Module](efficient-ps.md)
        - heart anomaly detection:
            - [gated_recurrent_unit Module](gated-recurrent-unit-learner.md)
            - [attention_neural_bag_of_feature_learner Module](attention-neural-bag-of-feature-learner.md)
        - fall detection:
            - [fall_detection Module](fall-detection.md)

    - `control` Module
        - [mobile_manipulation Module](mobile-manipulation.md)
        - [single_demo_grasp Module](single-demonstration-grasping.md)

    - `simulation` Module
        - [human_model_generation Module](human_model_generation.md)
    - `data_generation` Module
        - [synthetic_facial_image_generation Module](synthetic_facial_image_generator.md)
        - [human_model_generation Module](human-model-generation.md)
    - `utils` Module
        - [Hyperparameter Tuning Module](hyperparameter_tuner.md)
        - [Ambiguity Measure Module](ambiguity_measure.md)
    - `Stand-alone Utility Frameworks`
        - [Engine Agnostic Gym Environment with Reactive extension (EAGERx)](eagerx.md)
- [ROS Bridge Package](opendr-ros-bridge.md)
- [C Inference API](c-api.md)
    - [data.h](c-data-h.md)
    - [target.h](c-target-h.md)
    - [opendr_utils.h](c-opendr-utils-h.md)
    - [face_recognition.h](c-face-recognition-h.md)
- Projects
    - `C API` Module
        - [face recognition Demo](/projects/c_api)
    - `control` Module
        - [mobile_manipulation Demo](/projects/python/control/mobile_manipulation)
        - [single_demo_grasp Demo](/projects/python/control/single_demo_grasp)
    - `opendr workspace` Module
        - [opendr_ws](/projects/opendr_ws)
    - `perception` Module
        - activity recognition:
            - [activity_recognition Demo](/projects/python/perception/activity_recognition/demos/online_recognition)
        - face recognition:
            - [face_recognition_Demo](/projects/python/perception/face_recognition)
        - facial expression recognition:
            - [landmark_based_facial_expression_recognition Demo](/projects/python/perception/facial_expression_recognition/landmark_based_facial_expression_recognition)
            - [image_based_facial_emotion_estimation Demo](/projects/python/perception/facial_expression_recognition/image_based_facial_emotion_estimation)
        - heart anomaly detection:
            - [heart anomaly detection Demo](/projects/python/perception/heart_anomaly_detection)
        - pose estimation:
            - [lightweight_open_pose Demo](/projects/python/perception/pose_estimation/lightweight_open_pose)
            - [high_resolution_pose_estimation Demo](/projects/python/perception/pose_estimation/high_resolution_pose_estimation)
        - multimodal human centric:
            - [rgbd_hand_gesture_learner Demo](/projects/python/perception/multimodal_human_centric/rgbd_hand_gesture_recognition)
            - [audiovisual_emotion_recognition Demo](/projects/python/perception/multimodal_human_centric/audiovisual_emotion_recognition)
        - object detection 2d:
            - [nanodet Demo](/projects/python/perception/object_detection_2d/nanodet)
            - [detr Demo](/projects/python/perception/object_detection_2d/detr)
            - [gem Demo](/projects/python/perception/object_detection_2d/gem)
            - [retinaface Demo](/projects/python/perception/object_detection_2d/retinaface)
            - [centernet Demo](/projects/python/perception/object_detection_2d/centernet)
            - [ssd Demo](/projects/python/perception/object_detection_2d/ssd)
            - [yolov3 Demo](/projects/python/perception/object_detection_2d/yolov3)
              [yolov5 Demo](/projects/python/perception/object_detection_2d/yolov5)
            - [seq2seq-nms Demo](/projects/python/perception/object_detection_2d/nms/seq2seq-nms)
        - object detection 3d:
            - [voxel Demo](/projects/python/perception/object_detection_3d/demos/voxel_object_detection_3d)
        - object tracking 2d:
            - [fair_mot Demo](/projects/python/perception/object_tracking_2d/demos/fair_mot_deep_sort)
            - [siamrpn Demo](/projects/python/perception/object_tracking_2d/demos/siamrpn)
        - panoptic segmentation:
            - [efficient_ps Demo](/projects/python/perception/panoptic_segmentation/efficient_ps)
        - semantic segmentation:
            - [bisnet Demo](/projects/python/perception/semantic_segmentation/bisenet)
        - action recognition:
            - [skeleton_based_action_recognition Demo](/projects/python/perception/skeleton_based_action_recognition)
        - fall detection:
            - [fall_detection Demo](/projects/python/perception/fall_detection.md)
        - [full_map_posterior_slam Module](/projects/python/perception/slam/full_map_posterior_gmapping)
    - `simulation` Module
        - [SMPL+D Human Models Dataset](/projects/python/simulation/SMPL%2BD_human_models)
        - [Human-Data-Generation-Framework](/projects/python/simulation/human_dataset_generation)
        - [Human Model Generation Demos](/projects/python/simulation/human_dataset_generation)
    - `utils` Module
        - [Hyperparameter Tuning Module](/projects/python/utils/hyperparameter_tuner)
- [Known Issues](/docs/reference/issues.md)
