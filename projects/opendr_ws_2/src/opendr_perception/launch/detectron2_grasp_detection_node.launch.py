import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('opendr_perception'),
        'resource',
        'params.yaml'
        )

    node = Node(
        package='opendr_perception',
        name='detectron2_grasp_detection_node',
        namespace='opendr',
        executable='detectron2_grasp_detection_node',
        parameters=[config]
    )

    ld.add_action(node)
    return ld
