cmake_minimum_required(VERSION 2.8.12)
project(mobile_manipulation_rl_demo)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        )

####################################
### catkin specific configuration ##
####################################
### The catkin_package macro generates cmake config files for your package
### Declare things to be passed to dependent projects
### INCLUDE_DIRS: uncomment this if your package contains header files
### LIBRARIES: libraries you create in this project that dependent projects also need
### CATKIN_DEPENDS: catkin_packages dependent projects also need
### DEPENDS: system dependencies of this project that dependent projects also need
#catkin_package(
#        INCLUDE_DIRS include
#        LIBRARIES mobile_manipulation_rl alglib gripper_planner
#        CATKIN_DEPENDS roscpp rospy std_msgs pybind11_catkin
#        # tmc_robot_kinematics_model
#        # DEPENDS system_lib
#)
#
############
### Build ##
############
#
### Specify additional locations of header files
### Your package locations should be listed before other locations
#include_directories(
#        include
#        ${catkin_INCLUDE_DIRS}
#        ${OpenCV3_INCLUDE_DIRS}
#)
#
### Declare a C++ library
#
#add_library(myutils src/utils.cpp)
#target_link_libraries(myutils yaml-cpp ${catkin_LIBRARIES})
#
#add_library(gaussian_mixture_model src/gripper_planner/gaussian_mixture_model.cpp)
#target_link_libraries(gaussian_mixture_model myutils ${catkin_LIBRARIES})
#
#add_library(base_gripper_planner include/gripper_planner/base_gripper_planner.h src/gripper_planner/base_gripper_planner.cpp)
#target_link_libraries(base_gripper_planner myutils ${catkin_LIBRARIES})
#
#add_library(linear_planner src/gripper_planner/linear_planner.cpp)
#target_link_libraries(linear_planner base_gripper_planner myutils ${catkin_LIBRARIES})
#
#add_library(gmm_planner src/gripper_planner/gmm_planner.cpp)
#target_link_libraries(gmm_planner base_gripper_planner myutils ${catkin_LIBRARIES})
#
#add_library(worlds src/worlds.cpp)
#target_link_libraries(worlds myutils ${catkin_LIBRARIES})
#
#add_library(robot_env src/robot_env.cpp)
#target_link_libraries(robot_env worlds myutils ${LIBGP_LIBRARIES} ${catkin_LIBRARIES})
#
#add_library(robot_pr2 src/robot_pr2.cpp)
#target_link_libraries(robot_pr2 worlds myutils ${catkin_LIBRARIES})
#
#add_library(robot_tiago src/robot_tiago.cpp)
#target_link_libraries(robot_tiago worlds myutils ${catkin_LIBRARIES})
#
##add_library(robot_hsr src/robot_hsr.cpp)
##target_link_libraries(robot_hsr worlds myutils ${catkin_LIBRARIES})
#
## pybind
#pybind_add_module(pybindings SHARED src/worlds src/robot_env src/pybindings src/robot_pr2
#        src/robot_tiago src/utils src/gripper_planner/base_gripper_planner src/gripper_planner/linear_planner src/gripper_planner/gmm_planner
#        src/gripper_planner/gaussian_mixture_model)
##src/alglib/alglibinternal src/alglib/alglibmisc src/alglib/ap src/alglib/dataanalysis src/alglib/diffequations src/alglib/fasttransforms src/alglib/integration src/alglib/interpolation src/alglib/linalg src/alglib/optimization src/alglib/solvers src/alglib/specialfunctions src/alglib/statistics)
#target_link_libraries(pybindings PRIVATE worlds robot_env robot_pr2 robot_tiago myutils
#        base_gripper_planner linear_planner gmm_planner gaussian_mixture_model
#        ${LIBGP_LIBRARIES} ${catkin_LIBRARIES}
#        )
