# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup

package_name = 'opendr_data_generation'

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
    description='OpenDR\'s ROS2 nodes for data generation package',
    license='Apache License v2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'synthetic_facial_generation = opendr_data_generation.synthetic_facial_generation_node:main'
        ],
    },
)
