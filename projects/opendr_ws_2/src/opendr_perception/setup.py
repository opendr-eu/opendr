from setuptools import setup

package_name = 'opendr_perception'

setup(
    name=package_name,
    version='1.0.0',
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
            'object_detection_2d_centernet = opendr_perception.object_detection_2d_centernet_node:main',
            'object_detection_2d_detr = opendr_perception.object_detection_2d_detr_node:main',
            'object_detection_2d_yolov3 = opendr_perception.object_detection_2d_yolov3_node:main',
            'object_detection_2d_ssd = opendr_perception.object_detection_2d_ssd_node:main',
            'face_detection_retinaface = opendr_perception.face_detection_retinaface_node:main',
            'semantic_segmentation_bisenet = opendr_perception.semantic_segmentation_bisenet_node:main',
            'panoptic_segmentation = opendr_perception.panoptic_segmentation_efficient_ps:main',
            'face_recognition = opendr_perception.face_recognition_node:main',
            'fall_detection = opendr_perception.fall_detection_node:main',
            'video_activity_recognition = opendr_perception.video_activity_recognition_node:main',
        ],
    },
)
