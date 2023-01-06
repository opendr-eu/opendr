// Copyright 2020-2023 OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MOBILE_MANIPULATION_RL_ROBOT_ENV_H
#define MOBILE_MANIPULATION_RL_ROBOT_ENV_H

#include <eigen_conversions/eigen_msg.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <moveit_msgs/RobotState.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <std_srvs/Empty.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/system/error_code.hpp>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include "std_msgs/ColorRGBA.h"
#include "tf/transform_datatypes.h"
#include "visualization_msgs/MarkerArray.h"

#include "mobile_manipulation_rl/utils.hpp"
#include "mobile_manipulation_rl/worlds.hpp"

struct RobotState {
  tf::Transform base_tf;              // in world frame
  tf::Transform gripper_tf;           // in world frame
  tf::Transform relative_gripper_tf;  // in base frame
  std::vector<double> joint_values;

  RobotObs toRobotObs(RobotState prev_robot_state, bool ik_fail, double reward, bool done) {
    tf::Vector3 base_velocity = base_tf.getOrigin() - prev_robot_state.base_tf.getOrigin();
    double roll, pitch, yaw, yaw2;
    tf::Matrix3x3(base_tf.getRotation()).getRPY(roll, pitch, yaw);
    tf::Matrix3x3(prev_robot_state.base_tf.getRotation()).getRPY(roll, pitch, yaw2);
    double base_rotation_velocity = utils::rpyAngleDiff(yaw, yaw2);
    tf::Transform gripper_velocities(utils::calcDq(prev_robot_state.gripper_tf.getRotation(), gripper_tf.getRotation()),
                                     gripper_tf.getOrigin() - prev_robot_state.gripper_tf.getOrigin());
    return RobotObs{utils::tfToList(base_tf, true),
                    utils::tfToList(gripper_tf, true),
                    utils::tfToList(relative_gripper_tf, true),
                    joint_values,
                    utils::vector3ToList(base_velocity),
                    base_rotation_velocity,
                    utils::tfToList(gripper_velocities, true),
                    ik_fail,
                    reward,
                    done};
  }
};

// helper to be able to call ros::init before initialising node handle and rate
class ROSCommonNode {
protected:
  ROSCommonNode(int argc, char **argv, const char *node_name) { ros::init(argc, argv, node_name); }
};

class RobotEnv : ROSCommonNode {
private:
  ros::Publisher gripper_visualizer_;
  ros::Publisher robstate_visualizer_;
  ros::Publisher base_cmd_pub_;
  ros::ServiceClient client_get_scene_;
  visualization_msgs::MarkerArray gripper_plan_marker_;

  ros::AsyncSpinner *spinner_;

  std::vector<PathPoint> pathPoints_;
  bool verbose_;

  int marker_counter_ = 0;
  double time_ = 0;
  double updateTime();

  const double time_step_train_;

  // For collision checking
  robot_state::GroupStateValidityCallbackFn constraint_callback_fn_;
  //    collision_detection::AllowedCollisionMatrix acm_;
  //    void setAllowedCollisionMatrix(planning_scene::PlanningScenePtr planning_scene, std::vector<std::string> obj_names, bool
  //    allow); bool check_scene_collisions();

  bool setInitialGripperPose(const std::string &initial_joint_distribution);
  void setInitialPose(const std::vector<double> &initial_base_pose, const std::string &initial_joint_distribution);
  //    void update_current_gripper_from_world();
  tf::Vector3 worldToRelativeEEVelocities(const tf::Vector3 &ee_vel_world) const;

protected:
  ros::NodeHandle *nh_;
  std::vector<std::string> joint_names_;
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
  planning_scene::PlanningScenePtr planning_scene_;
  ros::Rate rate_;
  random_numbers::RandomNumberGenerator rng_;
  // simulator we use
  BaseWorld *world_;
  // robot specific values
  std::string robo_conf_path_;
  RoboConf robo_config_;
  // current state of the robot
  RobotState prev_robot_state_;
  RobotState robot_state_;

  robot_state::RobotStatePtr kinematic_state_;
  robot_state::JointModelGroup *joint_model_group_;
  const std::string strategy_;
  const bool init_controllers_;
  const double penalty_scaling_;
  bool perform_collision_check_;
  double success_thres_dist_;
  double success_thres_rot_;

  virtual geometry_msgs::Twist calcDesiredBaseTransform(std::vector<double> &base_actions, tf::Vector3 planned_gripper_vel_rel,
                                                        tf::Transform &desired_base_tf, double transition_noise_base,
                                                        double &regularization, double const &last_dt) = 0;
  virtual bool findIk(const Eigen::Isometry3d &desired_state_eigen, const tf::Transform &desiredGripperTfWorld);
  virtual double calcReward(bool found_ik, double regularization);
  virtual void sendArmCommand(const std::vector<double> &target_joint_values, double exec_duration) = 0;
  virtual bool getArmSuccess() = 0;
  void addTrajectoryPoint(const tf::Transform &desired_gripper_tf, bool found_ik);

public:
  RobotEnv(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
           double time_step_world, bool perform_collision_check, std::string node_handle_name, bool verbose,
           std::string robo_conf_path);

  ~RobotEnv() {
    delete nh_;
    // spinner_->stop();
    if (init_controllers_) {
      delete spinner_;
    }
    delete world_;
  }

  RobotObs reset(const std::vector<double> &initial_base_pose, const std::string &initial_joint_distribution,
                 bool do_close_gripper, double success_thres_dist, double success_thres_rot);

  RobotObs step(std::vector<double> base_actions, const std::vector<double> &ee_velocities_world, double transition_noise_base);

  // easiest way to know the dim without having to enforce that everything is already initialised
  int getObsDim() { return 21 + joint_names_.size(); };
  bool getInitControllers() const { return init_controllers_; };
  int getFirstSeed() { return (int)rng_.getFirstSeed(); };
  std::string getStrategy() const { return strategy_; };
  double getPenaltyScaling() const { return penalty_scaling_; };
  double getTimeStepWorld() const { return 1.0 / rate_.expectedCycleTime().toSec(); };
  bool getPerformCollisionCheck() const { return perform_collision_check_; };
  std::string getNodeHandleName() const { return nh_->getNamespace(); };
  bool getVerbose() const { return verbose_; };
  // pybind will complain if pure virtual here
  virtual void openGripper(double position, bool wait_for_result) { throw std::runtime_error("NOT IMPLEMENTED YET"); };
  virtual void closeGripper(double position, bool wait_for_result) { throw std::runtime_error("NOT IMPLEMENTED YET"); };
  void setWorld(const std::string &world_type);
  std::string getWorld() { return world_->name_; };
  bool isAnalyticalWorld() { return world_->isAnalytical(); };
  std::vector<double> tipToGripperTf(const std::vector<double> &tip_tf);
  std::vector<double> gripperToTipTf(const std::vector<double> &tip_tf);
  std::tuple<std::map<std::string, std::vector<double>>, std::map<std::string, std::string>> getRoboConfig() {
    return robo_config_.toPython();
  };
  std::string getRoboConfPath() { return robo_conf_path_; };
  RobotObs getRobotObs() { return robot_state_.toRobotObs(prev_robot_state_, false, 0.0, false); };
  void publishMarker(const std::vector<double> &marker_tf, int marker_id, const std::string &name_space,
                     const std::string &color, double alpha, const std::string &geometry,
                     const std::vector<double> &marker_scale);
  std::vector<double> worldToRelativeTf(const std::vector<double> &tf_world) const {
    return utils::tfToList(robot_state_.base_tf.inverse() * utils::listToTf(tf_world));
  }
};

class RobotOmniDrive : public RobotEnv {
protected:
  geometry_msgs::Twist calcDesiredBaseTransform(std::vector<double> &base_actions, tf::Vector3 planned_gripper_vel_rel,
                                                tf::Transform &desired_base_tf, double transition_noise_base,
                                                double &regularization, double const &last_dt) override;

public:
  RobotOmniDrive(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
                 double time_step, bool perform_collision_check, std::string node_handle_name, bool verbose,
                 std::string robo_conf_path) :
    RobotEnv(seed, strategy, world_type, init_controllers, penalty_scaling, time_step, perform_collision_check,
             node_handle_name, verbose, robo_conf_path){};
};

class RobotDiffDrive : public RobotEnv {
protected:
  geometry_msgs::Twist calcDesiredBaseTransform(std::vector<double> &base_actions, tf::Vector3 planned_gripper_vel_rel,
                                                tf::Transform &desired_base_tf, double transition_noise_base,
                                                double &regularization, double const &last_dt) override;

public:
  RobotDiffDrive(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
                 double time_step, bool perform_collision_check, std::string node_handle_name, bool verbose,
                 std::string robo_conf_path) :
    RobotEnv(seed, strategy, world_type, init_controllers, penalty_scaling, time_step, perform_collision_check,
             node_handle_name, verbose, robo_conf_path){};
};

namespace validity_fun {
  bool validityCallbackFn(planning_scene::PlanningScenePtr &planning_scene,
                          // const kinematics_constraint_aware::KinematicsRequest &request,
                          // kinematics_constraint_aware::KinematicsResponse &response,
                          const robot_state::RobotStatePtr &kinematic_state,
                          const robot_state::JointModelGroup *joint_model_group, const double *joint_group_variable_values
                          // const std::vector<double> &joint_group_variable_values
  );

}

#endif  // MOBILE_MANIPULATION_RL_ROBOT_ENV_H
