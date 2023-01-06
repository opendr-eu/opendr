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

#include <mobile_manipulation_rl/robot_pr2.hpp>

RobotPR2::RobotPR2(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
                   double time_step, bool perform_collision_check, std::string node_handle_name, bool verbose,
                   std::string robo_conf_path) :
  RobotOmniDrive(seed, strategy, world_type, init_controllers, penalty_scaling, time_step, perform_collision_check,
                 node_handle_name, verbose, robo_conf_path) {
  if (init_controllers_) {
    arm_client_ = new TrajClientPR2("r_arm_controller/joint_trajectory_action", true);
    while (!arm_client_->waitForServer(ros::Duration(5.0))) {
      ROS_INFO("Waiting for the r_arm_controller/joint_trajectory_action action server to come up");
    }

    // switch_controller_client_ =
    // nh_->serviceClient<pr2_mechanism_msgs::SwitchController>("/pr2_controller_manager/switch_controller"); not sure yet if
    // want to do this for real execution only or always
    gripper_client_ = new GripperClientPR2("r_gripper_controller/gripper_action", true);
    while (!gripper_client_->waitForServer(ros::Duration(5.0))) {
      ROS_INFO("Waiting for the r_gripper_controller/gripper_action action server to come up");
    }

    arm_goal_.trajectory.points.resize(1);
    arm_goal_.trajectory.joint_names.resize(joint_names_.size());
    arm_goal_.trajectory.points[0].positions.resize(joint_names_.size());
    arm_goal_.trajectory.points[0].velocities.resize(joint_names_.size());
  }
}

void RobotPR2::sendArmCommand(const std::vector<double> &target_joint_values, double exec_duration) {
  for (int i = 0; i < joint_names_.size(); i++) {
    arm_goal_.trajectory.joint_names[i] = joint_names_[i];
    arm_goal_.trajectory.points[0].positions[i] = target_joint_values[i];
    arm_goal_.trajectory.points[0].velocities[i] = 0.0;
    //        ROS_INFO("%s: %f")
  }

  // When to start the trajectory
  arm_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
  // To be reached x seconds after starting along the trajectory
  arm_goal_.trajectory.points[0].time_from_start = ros::Duration(exec_duration);
  // send off commands to run in parallel
  arm_client_->sendGoal(arm_goal_);
}

bool RobotPR2::getArmSuccess() {
  arm_client_->waitForResult(ros::Duration(10.0));
  if (arm_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
    ROS_WARN("The arm_client_ failed.");
    // throw std::runtime_error("The arm_client_ failed.");
    return false;
  } else {
    return true;
  }
}

// http://library.isr.ist.utl.pt/docs/roswiki/pr2_controllers(2f)Tutorials(2f)Moving(20)the(20)gripper.html
void RobotPR2::moveGripper(double position, double effort, bool wait_for_result) {
  pr2_controllers_msgs::Pr2GripperCommandGoal goal;
  goal.command.position = position;
  goal.command.max_effort = effort;
  gripper_client_->sendGoal(goal);

  if (wait_for_result) {
    gripper_client_->waitForResult(ros::Duration(5.0));
    if (gripper_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
      ROS_WARN("The gripper failed.");
  }
}

void RobotPR2::openGripper(double position, bool wait_for_result) {
  moveGripper(position, -1.0, wait_for_result);  // Do not limit effort (negative)
}

void RobotPR2::closeGripper(double position, bool wait_for_result) {
  moveGripper(position, 200.0, wait_for_result);  // Close gently
}
