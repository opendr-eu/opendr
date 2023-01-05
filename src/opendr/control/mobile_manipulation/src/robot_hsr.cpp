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

#include <mobile_manipulation_rl/robot_hsr.hpp>
using Eigen::Affine3d;
using Eigen::AngleAxisd;
using Eigen::Translation3d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::string;
using std::vector;
using tmc_manipulation_types::JointState;
using tmc_robot_kinematics_model::IKRequest;
using tmc_robot_kinematics_model::IKSolver;
using tmc_robot_kinematics_model::IRobotKinematicsModel;
using tmc_robot_kinematics_model::NumericIKSolver;
using tmc_robot_kinematics_model::Tarp3Wrapper;

namespace HSRIK {
  const uint32_t kMaxItr = 10000;
  const double kEpsilon = 0.001;
  const double kConvergeThreshold = 1e-6;
  const char *const kModelPath = "/opt/ros/melodic/share/hsrb_description/robots/hsrb4s.urdf";
}  // namespace HSRIK

RobotHSR::RobotHSR(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
                   double time_step, bool perform_collision_check, std::string node_handle_name, bool verbose,
                   std::string robo_conf_path, double ik_slack_dist, double ik_slack_rot_dist, bool sol_dist_reward) :
  RobotOmniDrive(seed, strategy, world_type, init_controllers, penalty_scaling, time_step, perform_collision_check,
                 node_handle_name, verbose, robo_conf_path),
  ik_slack_dist_{ik_slack_dist},
  ik_slack_rot_dist_{ik_slack_rot_dist},
  sol_dist_reward_{sol_dist_reward} {
  setup();
}

void RobotHSR::setup() {
  if (perform_collision_check_) {
    throw std::runtime_error("findIk() not adapted for HSR yet");
  }

  // analytic solver don't need robot model.
  IRobotKinematicsModel::Ptr robot;

  // load robot model.
  std::string xml_string;

  std::fstream xml_file(HSRIK::kModelPath, std::fstream::in);
  while (xml_file.good()) {
    std::string line;
    std::getline(xml_file, line);
    xml_string += (line + "\n");
  }
  xml_file.close();

  robot.reset(new Tarp3Wrapper(xml_string));
  // https://git.hsr.io/koji_terada/example_hsr_ik/-/blob/master/src/example_numeric_ik.cpp
  numeric_solver_.reset(
    new NumericIKSolver(IKSolver::Ptr(), robot, HSRIK::kMaxItr, HSRIK::kEpsilon, HSRIK::kConvergeThreshold));

  // ik joints. analytic ik have to use these joint.
  ik_joint_names_.push_back("arm_lift_joint");
  ik_joint_names_.push_back("arm_flex_joint");
  ik_joint_names_.push_back("arm_roll_joint");
  ik_joint_names_.push_back("wrist_flex_joint");
  ik_joint_names_.push_back("wrist_roll_joint");
  ik_joint_names_.push_back("wrist_ft_sensor_frame_joint");

  if (init_controllers_) {
    std::vector<string> controllers_to_await;

    arm_client_ = new TrajClientHSR("/hsrb/arm_trajectory_controller/follow_joint_trajectory", true);
    controllers_to_await.push_back("arm_trajectory_controller");

    while (!arm_client_->waitForServer(ros::Duration(5.0))) {
      ROS_INFO("Waiting for the /hsrb/arm_trajectory_controller/follow_joint_trajectory action server to come up");
    }

    // do not check if running, see https://qa.hsr.io/en/question/2164/gripper_controller-not-present-in-simulation/
    gripper_client_ = new TrajClientHSR("/hsrb/gripper_controller/follow_joint_trajectory", true);
    // controllers_to_await.push_back("gripper_controller");
    // while(!gripper_client_->waitForServer(ros::Duration(5.0))){
    //     ROS_INFO("Waiting for the /hsrb/gripper_controller/follow_joint_trajectory action server to come up");
    // }

    arm_goal_.trajectory.points.resize(1);
    arm_goal_.trajectory.joint_names.resize(joint_names_.size() - 1);
    arm_goal_.trajectory.points[0].positions.resize(joint_names_.size() - 1);
    arm_goal_.trajectory.points[0].velocities.resize(joint_names_.size() - 1);

    // make sure the controller is running
    ros::ServiceClient controller_manager_client =
      nh_->serviceClient<controller_manager_msgs::ListControllers>("/hsrb/controller_manager/list_controllers");
    controller_manager_msgs::ListControllers list_controllers;

    while (!controller_manager_client.call(list_controllers)) {
      ROS_INFO("Waiting for /hsrb/controller_manager/list_controllers");
      ros::Duration(0.5).sleep();
    }

    std::string cname;
    for (int j = 0; j < controllers_to_await.size(); j++) {
      cname = controllers_to_await.back();
      controllers_to_await.pop_back();
      bool running = false;
      while (!running) {
        ROS_INFO_STREAM("Waiting for /hsrb/" << cname);
        ros::Duration(0.5).sleep();
        if (controller_manager_client.call(list_controllers)) {
          for (const auto &c : list_controllers.response.controller) {
            if (c.name == cname && c.state == "running") {
              running = true;
            }
          }
        }
      }
    }
  }
}

void RobotHSR::setIkSlack(double ik_slack_dist, double ik_slack_rot_dist) {
  ik_slack_dist_ = ik_slack_dist;
  ik_slack_rot_dist_ = ik_slack_rot_dist;
}

bool RobotHSR::findIk(const Eigen::Isometry3d &desired_state_eigen, const tf::Transform &desiredGripperTfWorld) {
  // *** make request for IK ***
  // useing base DOF as planar movement. analytic IK have to use kPlanar.
  IKRequest req(tmc_manipulation_types::kNone);
  // reference frame. analytic IK have to use hand_palm_link
  req.frame_name = robo_config_.global_link_transform;
  // offset from reference frame.
  req.frame_to_end = Affine3d::Identity();

  req.initial_angle.name = ik_joint_names_;
  // reference joint angles
  req.initial_angle.position.resize(6);
  req.initial_angle.position << robot_state_.joint_values[0], robot_state_.joint_values[1], robot_state_.joint_values[2],
    robot_state_.joint_values[3], robot_state_.joint_values[4], robot_state_.joint_values[5];
  req.use_joints = ik_joint_names_;
  Eigen::VectorXd weight_vector;
  // weight of joint angle. #1-3 weights are for base DOF.
  weight_vector.resize(8);
  weight_vector << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

  req.weight = weight_vector;
  req.origin_to_base = Affine3d::Identity();
  // reference positon.
  req.ref_origin_to_end = desired_state_eigen;

  // output values.
  JointState solution;
  Eigen::Affine3d origin_to_hand_result;
  // Eigen::Affine3d origin_to_base_solution;
  tmc_robot_kinematics_model::IKResult result;

  // Solve.
  result = numeric_solver_->Solve(req, solution,
                                  // origin_to_base_solution,
                                  origin_to_hand_result);

  kinematic_state_->setJointGroupPositions(joint_model_group_, solution.position);

  // Due to limit arm capabilities for most poses it will not be possible to find exact solution. Therefore allow a bit variance
  const Eigen::Affine3d &solution_state = kinematic_state_->getGlobalLinkTransform(robo_config_.global_link_transform);
  tf::Transform solution_state_tf, desiredState_tf;
  tf::transformEigenToTF(solution_state, solution_state_tf);
  tf::transformEigenToTF(desired_state_eigen, desiredState_tf);

  dist_solution_desired_ = (solution_state_tf.getOrigin() - desiredState_tf.getOrigin()).length();
  rot_dist_solution_desired_ = utils::calcRotDist(solution_state_tf, desiredState_tf);
  // std::cout << "success: " << (result == tmc_robot_kinematics_model::kSuccess) << ", dist_solution_desired_: " <<
  // dist_solution_desired_ << ", rot_dist_solution_desired_: " << rot_dist_solution_desired_ << std::endl;

  if (ik_slack_dist_ == 0.0) {
    return result == tmc_robot_kinematics_model::kSuccess;
  } else {
    // Due to the kinematics an exact solution is not possible in most situations
    // make slightly stricter than success_thres_dist_ as numeric error might cause it to fail to terminate if it finishes with
    // an error margin of success_thres_dist_ NOTE: RISK OF GETTING STUCK IF success_thres_dist_ < ik_slack_dist_!
    double dist_desired_goal = (desiredGripperTfWorld.getOrigin() - gripper_goal_wrist_.getOrigin()).length();
    if ((success_thres_dist_ < ik_slack_dist_) && (dist_desired_goal < 0.01)) {
      // enforce to achieve the final goal irrespective of the slack we give it
      return (dist_solution_desired_ < success_thres_dist_ && rot_dist_solution_desired_ < success_thres_rot_);
    } else {
      return (dist_solution_desired_ < ik_slack_dist_ && rot_dist_solution_desired_ < ik_slack_rot_dist_);
    }
  }
}

double RobotHSR::calcReward(bool found_ik, double regularization) {
  double reward = RobotEnv::calcReward(found_ik, regularization);

  if (sol_dist_reward_ && found_ik) {
    // scale to be a max of penalty_scaling_ * (-0.5 - 0.5)
    double dist_penalty = 0.5 * pow(dist_solution_desired_, 2) / pow(ik_slack_dist_, 2) +
                          0.5 * pow(rot_dist_solution_desired_, 2) / pow(ik_slack_rot_dist_, 2);
    reward -= dist_penalty;
  }

  return reward;
}

void RobotHSR::sendArmCommand(const std::vector<double> &target_joint_values, double exec_duration) {
  int j = 0;
  for (int i = 0; i < joint_names_.size(); i++) {
    // std::cout << joint_names_[i] << ": " << target_joint_values[i] << std::endl;
    // part of the moveit controller definition, but not part of
    // /opt/ros/melodic/share/hsrb_common_config/params/hsrb_controller_config.yaml
    if (joint_names_[i] != "wrist_ft_sensor_frame_joint") {
      arm_goal_.trajectory.joint_names[j] = joint_names_[i];
      arm_goal_.trajectory.points[0].positions[j] = target_joint_values[i];
      arm_goal_.trajectory.points[0].velocities[j] = 0.0;
      j++;
    }
  }

  // When to start the trajectory. Will get dropped with 0
  arm_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.05);
  // To be reached x seconds after starting along the trajectory
  arm_goal_.trajectory.points[0].time_from_start = ros::Duration(exec_duration);
  // send off commands to run in parallel
  arm_client_->sendGoal(arm_goal_);
}

bool RobotHSR::getArmSuccess() {
  arm_client_->waitForResult(ros::Duration(10.0));
  if (arm_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
    ROS_WARN("The arm_client_ failed.");
    // throw std::runtime_error("The arm_client_ failed.");
    return false;
  } else {
    return true;
  }
}

void RobotHSR::openGripper(double position, bool wait_for_result) {
  // hsr takes 1.0 as completely open -> calculate proportional to an assumed max. opening of 0.1m
  position = std::min(position / 0.1, 1.0);

  control_msgs::FollowJointTrajectoryGoal goal;
  goal.trajectory.joint_names.push_back("hand_motor_joint");

  goal.trajectory.points.resize(1);
  goal.trajectory.points[0].positions.resize(1);
  goal.trajectory.points[0].effort.resize(1);
  goal.trajectory.points[0].velocities.resize(1);

  goal.trajectory.points[0].positions[0] = position;
  goal.trajectory.points[0].velocities[0] = 0.0;
  goal.trajectory.points[0].effort[0] = 500;
  goal.trajectory.points[0].time_from_start = ros::Duration(3.0);

  // send message to the action server
  gripper_client_->sendGoal(goal);

  if (wait_for_result) {
    gripper_client_->waitForResult(ros::Duration(5.0));

    if (gripper_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
      ROS_WARN("The gripper controller failed.");
  }
}

void RobotHSR::closeGripper(double position, bool wait_for_result) {
  // 0.0 is not completely closed, but rather both 'forks' pointing straight ahead. -0.02 is roughly fully closed
  openGripper(position - 0.02, wait_for_result);
}
