from setuptools import setup

package_name = 'opendr_perception'

setup(
    name=package_name,
    version='2.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenDR Project Coordinator',
    maintainer_email='tefas@csd.auth.gr',
    description='OpenDR ROS2 nodes for the perception package',
    license='Apache License v2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimation = opendr_perception.pose_estimation_node:main',
            'hr_pose_estimation = opendr_perception.hr_pose_estimation_node:main',
            'object_detection_2d_centernet = opendr_perception.object_detection_2d_centernet_node:main',
            'object_detection_2d_detr = opendr_perception.object_detection_2d_detr_node:main',
            'object_detection_2d_yolov3 = opendr_perception.object_detection_2d_yolov3_node:main',
            'object_detection_2d_yolov5 = opendr_perception.object_detection_2d_yolov5_node:main',
            'object_detection_2d_ssd = opendr_perception.object_detection_2d_ssd_node:main',
            'object_detection_2d_nanodet = opendr_perception.object_detection_2d_nanodet_node:main',
            'object_detection_2d_gem = opendr_perception.object_detection_2d_gem_node:main',
            'object_tracking_2d_siamrpn = opendr_perception.object_tracking_2d_siamrpn_node:main',
            'face_detection_retinaface = opendr_perception.face_detection_retinaface_node:main',
            'semantic_segmentation_bisenet = opendr_perception.semantic_segmentation_bisenet_node:main',
            'panoptic_segmentation = opendr_perception.panoptic_segmentation_efficient_ps_node:main',
            'face_recognition = opendr_perception.face_recognition_node:main',
            'fall_detection = opendr_perception.fall_detection_node:main',
            'point_cloud_dataset = opendr_perception.point_cloud_dataset_node:main',
            'image_dataset = opendr_perception.image_dataset_node:main',
            'object_detection_3d_voxel = opendr_perception.object_detection_3d_voxel_node:main',
            'object_tracking_3d_ab3dmot = opendr_perception.object_tracking_3d_ab3dmot_node:main',
            'object_tracking_2d_fair_mot = opendr_perception.object_tracking_2d_fair_mot_node:main',
            'object_tracking_2d_deep_sort = opendr_perception.object_tracking_2d_deep_sort_node:main',
            'video_activity_recognition = opendr_perception.video_activity_recognition_node:main',
            'audiovisual_emotion_recognition = opendr_perception.audiovisual_emotion_recognition_node:main',
            'speech_command_recognition = opendr_perception.speech_command_recognition_node:main',
            'heart_anomaly_detection = opendr_perception.heart_anomaly_detection_node:main',
            'rgbd_hand_gestures_recognition = opendr_perception.rgbd_hand_gesture_recognition_node:main',
            'landmark_based_facial_expression_recognition = \
            opendr_perception.landmark_based_facial_expression_recognition_node:main',
            'facial_emotion_estimation = opendr_perception.facial_emotion_estimation_node:main',
            'skeleton_based_action_recognition = opendr_perception.skeleton_based_action_recognition_node:main',
            'continual_skeleton_based_action_recognition = \
            opendr_perception.continual_skeleton_based_action_recognition_node:main',
        ],
    },
)
