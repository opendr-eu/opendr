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

#include <mobile_manipulation_rl/robot_tiago.hpp>

RobotTiago::RobotTiago(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers,
                       double penalty_scaling, double time_step, bool perform_collision_check, std::string node_handle_name,
                       bool verbose, std::string robo_conf_path) :
  RobotDiffDrive(seed, strategy, world_type, init_controllers, penalty_scaling, time_step, perform_collision_check,
                 node_handle_name, verbose, robo_conf_path) {
  setup();
}

void RobotTiago::setup() {
  if (init_controllers_) {
    arm_client_.reset(new TrajClientTiago("/arm_controller/follow_joint_trajectory"));
    while (!arm_client_->waitForServer(ros::Duration(5.0))) {
      ROS_INFO("Waiting for the arm_controller/follow_joint_trajectory action server to come up");
    }

    torso_client_.reset(new TrajClientTiago("/torso_controller/follow_joint_trajectory"));
    while (!torso_client_->waitForServer(ros::Duration(5.0))) {
      ROS_INFO("Waiting for the torso_controller/follow_joint_trajectory action server to come up");
    }

    arm_goal_.trajectory.joint_names.resize(joint_names_.size() - 1);
    arm_goal_.trajectory.points.resize(1);
    arm_goal_.trajectory.points[0].positions.resize(joint_names_.size() - 1);
    arm_goal_.trajectory.points[0].velocities.resize(joint_names_.size() - 1);

    torso_goal_.trajectory.joint_names.resize(1);
    torso_goal_.trajectory.points.resize(1);
    torso_goal_.trajectory.points[0].positions.resize(1);
    torso_goal_.trajectory.points[0].velocities.resize(1);

    gripper_client_.reset(new TrajClientTiago("/gripper_controller/follow_joint_trajectory"));
    while (!gripper_client_->waitForServer(ros::Duration(5.0))) {
      ROS_INFO("Waiting for the gripper_controller/follow_joint_trajectory action server to come up");
    }
  }
}

void RobotTiago::sendArmCommand(const std::vector<double> &target_joint_values, double exec_duration) {
  // plan gripper and torso
  // joint_names_ for group arm_torso = [torso_lift_joint, arm_1_joint, arm_2_joint, arm_3_joint, arm_4_joint, arm_5_joint,
  // arm_6_joint, arm_7_joint]
  int idx;
  for (int i = 0; i < joint_names_.size(); i++) {
    // std::cout << joint_names_[i] << std::endl;
    if (joint_names_[i] == "torso_lift_joint") {
      idx = 0;
      torso_goal_.trajectory.joint_names[idx] = joint_names_[i];
      torso_goal_.trajectory.points[0].positions[idx] = target_joint_values[i];
      torso_goal_.trajectory.points[0].velocities[idx] = 0.0;
    } else {
      idx = i - 1;
      arm_goal_.trajectory.joint_names[idx] = joint_names_[i];
      arm_goal_.trajectory.points[0].positions[idx] = target_joint_values[i];
      arm_goal_.trajectory.points[0].velocities[idx] = 0.0;
    }
  }

  // When to start the trajectory
  arm_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
  torso_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
  // To be reached x seconds after starting along the trajectory
  arm_goal_.trajectory.points[0].time_from_start = ros::Duration(exec_duration);
  torso_goal_.trajectory.points[0].time_from_start = ros::Duration(exec_duration);
  // send off commands to run in parallel
  arm_client_->sendGoal(arm_goal_);
  torso_client_->sendGoal(torso_goal_);
}

bool RobotTiago::getArmSuccess() {
  bool success = true;
  torso_client_->waitForResult(ros::Duration(10.0));
  if (torso_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
    ROS_WARN("The torso_client_ failed.");
    success = false;
  }
  arm_client_->waitForResult(ros::Duration(10.0));
  if (arm_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
    ROS_WARN("The arm_client_ failed.");
    // throw std::runtime_error("The arm_client_ failed.");
    success &= false;
  }
  return success;
}

void RobotTiago::openGripper(double position, bool wait_for_result) {
  control_msgs::FollowJointTrajectoryGoal goal;

  // The joint names, which apply to all waypoints
  goal.trajectory.joint_names.push_back("gripper_left_finger_joint");
  goal.trajectory.joint_names.push_back("gripper_right_finger_joint");
  int n = goal.trajectory.joint_names.size();

  // Two waypoints in this goal trajectory
  goal.trajectory.points.resize(1);

  // First trajectory point
  // Positions
  int index = 0;
  goal.trajectory.points[index].positions.resize(n);
  goal.trajectory.points[index].positions[0] = position / 2;
  goal.trajectory.points[index].positions[1] = position / 2;
  // Velocities
  goal.trajectory.points[index].velocities.resize(n);
  for (int j = 0; j < n; ++j) {
    goal.trajectory.points[index].velocities[j] = 0.0;
  }
  goal.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
  goal.trajectory.points[index].time_from_start = ros::Duration(2.0);

  gripper_client_->sendGoal(goal);

  if (wait_for_result) {
    gripper_client_->waitForResult(ros::Duration(5.0));
    if (gripper_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
      ROS_WARN("The gripper client failed.");
    ros::Duration(0.1).sleep();
  }
}

void RobotTiago::closeGripper(double position, bool wait_for_result) {
  openGripper(position, wait_for_result);
}
