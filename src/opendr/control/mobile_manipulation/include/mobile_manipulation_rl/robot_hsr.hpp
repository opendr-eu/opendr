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

#ifndef MOBILE_MANIPULATION_RL_ROBOT_HSR_H
#define MOBILE_MANIPULATION_RL_ROBOT_HSR_H
#include <mobile_manipulation_rl/robot_env.hpp>

#include <tmc_robot_kinematics_model/numeric_ik_solver.hpp>
#include <tmc_robot_kinematics_model/robot_kinematics_model.hpp>
#include <tmc_robot_kinematics_model/tarp3_wrapper.hpp>

#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <control_msgs/FollowJointTrajectoryGoal.h>
#include <controller_manager_msgs/ControllerState.h>
#include <controller_manager_msgs/ListControllers.h>

typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> TrajClientHSR;

class RobotHSR : public RobotOmniDrive {
private:
  tmc_robot_kinematics_model::IKSolver::Ptr numeric_solver_;
  tmc_manipulation_types::NameSeq ik_joint_names_;

  double dist_solution_desired_;
  double rot_dist_solution_desired_;
  double ik_slack_dist_;
  double ik_slack_rot_dist_;
  bool sol_dist_reward_;
  tf::Transform gripper_goal_wrist_;

  TrajClientHSR *arm_client_;
  TrajClientHSR *gripper_client_;
  void setup();
  bool findIk(const Eigen::Isometry3d &desired_state_eigen, const tf::Transform &desiredGripperTfWorld) override;
  double calcReward(bool found_ik, double regularization) override;

  control_msgs::FollowJointTrajectoryGoal arm_goal_;
  void sendArmCommand(const std::vector<double> &target_joint_values, double exec_duration) override;
  bool getArmSuccess() override;

public:
  RobotHSR(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
           double time_step, bool perform_collision_check, std::string node_handle_name, bool verbose,
           std::string robo_conf_path, double ik_slack_dist, double ik_slack_rot_dist, bool sol_dist_reward);

  ~RobotHSR() {
    if (init_controllers_) {
      delete gripper_client_;
      delete arm_client_;
    }
  }

  void openGripper(double position, bool wait_for_result) override;
  void closeGripper(double position, bool wait_for_result) override;
  void setIkSlack(double ik_slack_dist, double ik_slack_rot_dist);
  void setGripperGoalWrist(const std::vector<double> &gripper_goal_wrist) {
    gripper_goal_wrist_ = utils::listToTf(gripper_goal_wrist);
  };
  double getIkSlackDist() const { return ik_slack_dist_; };
  double getIkSlackRotDist() const { return ik_slack_rot_dist_; };
  bool getSolDistReward() const { return sol_dist_reward_; };
};
#endif  // MOBILE_MANIPULATION_RL_ROBOT_HSR_H