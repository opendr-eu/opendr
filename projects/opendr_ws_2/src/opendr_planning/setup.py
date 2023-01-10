from setuptools import setup

package_name = 'opendr_planning'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/end_to_end_planning_robot_launch.py']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/train-no-dynamic-random-obstacles.wbt']))
data_files.append(('share/' + package_name + '/protos', ['protos/box.proto']))
data_files.append(('share/' + package_name + '/resource', ['resource/uav_robot.urdf']))
data_files.append(('share/' + package_name, ['package.xml']))


setup(
    name=package_name,
    version='2.1.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenDR Project Coordinator',
    maintainer_email='tefas@csd.auth.gr',
    description='OpenDR ROS2 nodes for the planning package',
    license='Apache License v2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'end_to_end_planner = opendr_planning.end_to_end_planner_node:main',
            'end_to_end_planning_robot_driver = opendr_planning.end_to_end_planning_robot_driver:main',
        ],
    },
)
