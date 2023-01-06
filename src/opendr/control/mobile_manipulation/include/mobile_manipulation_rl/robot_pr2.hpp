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

#ifndef MOBILE_MANIPULATION_RL_ROBOT_PR2_H
#define MOBILE_MANIPULATION_RL_ROBOT_PR2_H

#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/Twist.h>
#include <pr2_controllers_msgs/JointTrajectoryAction.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <pr2_mechanism_msgs/SwitchController.h>
#include <mobile_manipulation_rl/robot_env.hpp>

// Our Action interface type, provided as a typedef for convenience
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::Pr2GripperCommandAction> GripperClientPR2;
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::JointTrajectoryAction> TrajClientPR2;

class RobotPR2 : public RobotOmniDrive {
private:
  TrajClientPR2 *arm_client_;
  GripperClientPR2 *gripper_client_;
  // ros::ServiceClient switch_controller_client_;
  pr2_controllers_msgs::JointTrajectoryGoal arm_goal_;
  void sendArmCommand(const std::vector<double> &target_joint_values, double exec_duration) override;
  bool getArmSuccess() override;
  // void stop_controllers();
  // void start_controllers();
  void moveGripper(double position, double effort, bool wait_for_result);

public:
  RobotPR2(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
           double time_step, bool perform_collision_check, std::string node_handle_name, bool verbose,
           std::string robo_conf_path);

  ~RobotPR2() {
    if (init_controllers_) {
      delete gripper_client_;
      delete arm_client_;
    }
  }

  void openGripper(double position, bool wait_for_result) override;
  void closeGripper(double position, bool wait_for_result) override;
};
#endif  // MOBILE_MANIPULATION_RL_ROBOT_PR2_H