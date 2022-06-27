from setuptools import setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='test',
    maintainer_email='test@csd.auth.gr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimation = perception.pose_estimation_node:main',
            'object_detection_2d_centernet = perception.object_detection_2d_centernet_node:main',
            'object_detection_2d_detr = perception.object_detection_2d_detr_node:main',
            'object_detection_2d_yolov3 = perception.object_detection_2d_yolov3_node:main',
            'object_detection_2d_ssd = perception.object_detection_2d_ssd_node:main',
            'face_detection_retinaface = perception.face_detection_retinaface_node:main',
            'tester = perception.subscriber_tester:main',
        ],
    },
)
