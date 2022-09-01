# Full Map Posterior SLAM

Traditional *SLAM* algorithm for estimating a robot's position and a 2D, grid-based map of the environment from planar LiDAR scans.
Based on OpenSLAM GMapping, with additional functionality for computing the closed-form Full Map Posterior Distribution.

For more details on the launchers and tools, see the [FMP_Eval Readme](../../projects/python/perception/slam/full_map_posterior_gmapping/src/fmp_slam_eval/README.md).

For more details on the actual SLAM algorithm and its ROS node wrapper, see the [SLAM_GMapping Readme](../../projects/python/perception/slam/full_map_posterior_gmapping/src/slam_gmapping/README.md).

## Demo Usage
A demo ROSBag for a square corridor can be found in the Map Simulator submodule in `src/map_simulator/rosbags/`, as well as preconfigured ***roslaunch***
files in `src/fmp_gmapping/launch/` to start using it right away.

In order to start a launch file, with the environment properly set to the FMP_gmapping workspace, run:
```console
foo@bar:-$ roslaunch fmp_slam_eval experiment.launch
```

This will start the following processes and nodes:

* `roscore`: ROS Master process
* `gmapping` **node:** Main Grid-based SLAM node
* `rosbag play` **node:** Simulation node playing the rosbag for the demo 10Loop simulation
* `rviz`: Program for the visualisation of the step-by-step results

Other ROSBags can be easily generated with the map simulator script from either new custom scenarios, or from the test configuration files in `src/map_simulator/scenarios/robots/` directory.

For more information on how to define custom test scenarios and converting them to ROSBags, see the [Map_Simulator Readme](../../projects/python/perception/slam/full_map_posterior_gmapping/src/map_simulator/README.md).