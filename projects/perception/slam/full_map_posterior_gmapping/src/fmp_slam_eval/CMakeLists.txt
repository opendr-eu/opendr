cmake_minimum_required(VERSION 3.0.2)
project(fmp_slam_eval)

find_package(
        catkin REQUIRED COMPONENTS
        rospy std_msgs nav_msgs geometry_msgs sensor_msgs
)

catkin_python_setup()

catkin_package(
        INCLUDE_DIRS include
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
    PROGRAMS
        nodes/gt_mapping
        nodes/fmp_plot
        nodes/occ_map_saver
        nodes/odom_pose
        nodes/pose_error_calc
        nodes/err_collector
    DESTINATION
        ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
    DIRECTORY
        launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
install(
    DIRECTORY
        urdf/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)
install(
    DIRECTORY
        cfg/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg
)