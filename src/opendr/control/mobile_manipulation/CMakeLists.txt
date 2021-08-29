cmake_minimum_required(VERSION 2.8.12)
project(mobile_manipulation_rl)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        geometry_msgs
        cmake_modules
        roscpp
        tf
        tf_conversions
        eigen_conversions
        moveit_core
        moveit_ros_perception
        moveit_ros_planning_interface
        moveit_planners_ompl
        rosbag
        cmake_modules
        pybind11_catkin
        #  tmc_robot_kinematics_model
        )
find_package(Eigen REQUIRED)
find_package(cmake_modules REQUIRED)
find_package(yaml-cpp REQUIRED)
include_directories(
        include
        ${Eigen_INCLUDE_DIRS}
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
        INCLUDE_DIRS include
        LIBRARIES mobile_manipulation_rl alglib gripper_planner
        CATKIN_DEPENDS roscpp rospy std_msgs pybind11_catkin
        # tmc_robot_kinematics_model
        # DEPENDS system_lib
)

###########
## Build ##
###########
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV3_INCLUDE_DIRS}
)

add_library(myutils src/utils.cpp)
target_link_libraries(myutils yaml-cpp ${catkin_LIBRARIES})

add_library(gaussian_mixture_model src/gripper_planner/gaussian_mixture_model.cpp)
target_link_libraries(gaussian_mixture_model myutils ${catkin_LIBRARIES})

add_library(base_gripper_planner include/gripper_planner/base_gripper_planner.hpp src/gripper_planner/base_gripper_planner.cpp)
target_link_libraries(base_gripper_planner myutils ${catkin_LIBRARIES})

add_library(linear_planner src/gripper_planner/linear_planner.cpp)
target_link_libraries(linear_planner base_gripper_planner myutils ${catkin_LIBRARIES})

add_library(gmm_planner src/gripper_planner/gmm_planner.cpp)
target_link_libraries(gmm_planner base_gripper_planner myutils ${catkin_LIBRARIES})

add_library(worlds src/worlds.cpp)
target_link_libraries(worlds myutils ${catkin_LIBRARIES})

add_library(robot_env src/robot_env.cpp)
target_link_libraries(robot_env worlds myutils ${catkin_LIBRARIES})

add_library(robot_pr2 src/robot_pr2.cpp)
target_link_libraries(robot_pr2 worlds myutils ${catkin_LIBRARIES})

add_library(robot_tiago src/robot_tiago.cpp)
target_link_libraries(robot_tiago worlds myutils ${catkin_LIBRARIES})

#add_library(robot_hsr src/robot_hsr.cpp)
#target_link_libraries(robot_hsr worlds myutils ${catkin_LIBRARIES})

# pybind
pybind_add_module(pybindings SHARED src/worlds src/robot_env src/pybindings src/robot_pr2
        src/robot_tiago src/utils src/gripper_planner/base_gripper_planner src/gripper_planner/linear_planner src/gripper_planner/gmm_planner
        src/gripper_planner/gaussian_mixture_model)
target_link_libraries(pybindings PRIVATE worlds robot_env robot_pr2 robot_tiago myutils
        base_gripper_planner linear_planner gmm_planner gaussian_mixture_model
        ${catkin_LIBRARIES}
        )
