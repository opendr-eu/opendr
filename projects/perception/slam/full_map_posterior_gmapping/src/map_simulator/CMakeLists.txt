cmake_minimum_required(VERSION 3.0.2)
project(map_simulator)

find_package(catkin REQUIRED COMPONENTS
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
        scripts/mapsim2d.py

    DESTINATION
        ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
        DIRECTORY
            scenarios/
        DESTINATION
            ${CATKIN_PACKAGE_SHARE_DESTINATION}/scenarios
)
install(
        DIRECTORY
        launch/
        DESTINATION
        ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
install(
        DIRECTORY
            rosbags/
        DESTINATION
            ${CATKIN_PACKAGE_SHARE_DESTINATION}/rosbags
)