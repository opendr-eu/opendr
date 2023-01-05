from setuptools import setup

package_name = 'opendr_bridge'

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
    description='OpenDR ROS2 bridge package. This package provides a way to translate ROS2 messages into OpenDR' +
                'data types and vice versa.',
    license='Apache License v2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
