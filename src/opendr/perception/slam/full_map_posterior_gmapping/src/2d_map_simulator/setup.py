# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'map_simulator', 'map_simulator.enums',
        'map_simulator.geometry', 'map_simulator.geometry.primitives'
        'map_simulator.map_obstacles',
        'map_simulator.robot_commands',
        'map_simulator.robot_commands.message', 'map_simulator.robot_commands.misc', 'map_simulator.robot_commands.move'
    ],
    package_dir={
        '': 'src'
    },
)

setup(**setup_args)
