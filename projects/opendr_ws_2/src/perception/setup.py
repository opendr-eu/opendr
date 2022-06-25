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
            'panoptic_segmentation = perception.panoptic_segmentation_efficient_ps:main',
            'pose_estimation = perception.pose_estimation_node:main',
            'tester = perception.subscriber_tester:main',
        ],
    },
)
