# Full Map Posterior Grid-based Fast-SLAM

This project extends the OpenSLAM GMapping [***[1]***](#ref1)[***[2]***](#ref2) framework
and it's ROS wrapper [***[3]***](#ref3) to support computing Full Map Posteriors (besides the usual Most-Likely Map)
for the traditional Reflection Map and an additional Exponential Decay Rate Map models [***[4]***](#ref4).


## Modules

### Map Simulator
Utility for generating simulated data for testing and evaluating SLAM algorithms in the form of ROSBags.

For usage details, check the map_simulator [README](src/map_simulator/README.md) file.

### OpenSLAM GMapping
Libraries, utilities and executables for the standalone GMapping (i.e. outside of ROS), with the Full Map Posterior extended features included. 

### SLAM GMapping
Wrapper for the Full Map Posterior GMapping framework to be used inside the ROS environment.

For usage details, check the slam_gmapping [README](src/slam_gmapping/README.md) file.

### FMP SLAM Evaluation
Includes nodes and utilities for displaying and saving the generated maps, evaluating the pose error, and executing single or multiple test scenarios via launch files.

For usage details, check the fmp_slam_eval [README](src/fmp_slam_eval/README.md) file.

## Additional Datasets

Apart from creating custom test scenarios with the Map Simulator utility, common benchmark datasets can also be found in the 
following [repository](https://github.com/joseab10/slam_datasets).

In order to keep this toolbox as manageable as possible, they have to be downloaded independently.

The repository contains the following datasets, each of which has both a file with the raw data, and another one with the odometry corrected using the original implementation of GMapping.

* University of Freiburg
    * Campus
    * Building 79
    * Building 101
        
* Intel Research Lab

* MIT
    * Infinite Corridor
    * CSAIL

These datasets were collected from multiple sources and converted from their original CARMEN log files to ROSBags, and have been tested to work with FMP GMapping.

## Demo Usage
demo ROSBag for a square corridor can be found in the Map Simulator submodule in `src/map_simulator/rosbags/`, as well as preconfigured ***roslaunch***
files in `src/fmp_gmapping/launch/` to start using it right away.

In order to start a launch file, with the environment properly set to the FMP_gmapping workspace, run:
```console
foo@bar:-$ roslaunch fmp_slam_eval experiment.launch
```

This will start the following processes and nodes:

  * ***roscore:*** ROS Master process
  * ***gmapping*** **node:** Main Grid-based SLAM node
  * ***rosbag play*** **node:** Simulation node playing the rosbag for the demo 10Loop simulation 
  * ***rviz:*** Program for the visualisation of the step-by-step results 

## References
* <a name="ref1">***[1]***</a> G. Grisetti, C. Stachniss, and W. Burgard. [*Improved techniques for grid mapping with Rao-Blackwellized particle filters*](http://ais.informatik.uni-freiburg.de/publications/papers/grisetti07tro.pdf). IEEE Transactions on Robotics, 23(1):34?46, 2007.
* <a name="ref2">***[2]***</a> OpenSLAM Gmapping [source code](https://github.com/OpenSLAM-org/openslam_gmapping)
* <a name="ref3">***[3]***</a> ROS Perception SLAM Gmapping [source code](https://github.com/ros-perception/slam_gmapping)
* <a name="ref4">***[4]***</a> L. Luft, A. Schaefer, T. Schubert and W. Burgard. [*Closed-Form Full Map Posteriors for Robot Localization with LiDAR Sensors*](https://arxiv.org/abs/1910.10493). 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Sep 2017.