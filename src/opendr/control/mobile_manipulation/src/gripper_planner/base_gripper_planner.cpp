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

#include <gripper_planner/base_gripper_planner.hpp>

BaseGripperPlanner::BaseGripperPlanner(const std::vector<double> &gripper_goal_wrist,
                                       const std::vector<double> &initial_gripper_tf, const std::vector<double> &base_goal,
                                       const std::vector<double> &initial_base_tf, double success_thres_dist,
                                       double success_thres_rot, const double &min_planner_velocity,
                                       const double &max_planner_velocity, const double &slow_down_factor,
                                       const double &head_start, const double &time_step_train, const bool &is_analytic_env) :
  gripper_goal_wrist_{utils::listToTf(gripper_goal_wrist)},
  initial_gripper_tf_{utils::listToTf(initial_gripper_tf)},
  base_goal_{utils::listToTf(base_goal)},
  initial_base_tf_{utils::listToTf(initial_base_tf)},
  success_thres_dist_{success_thres_dist},
  success_thres_rot_{success_thres_rot},
  min_planner_velocity_{min_planner_velocity},
  max_planner_velocity_{max_planner_velocity},
  slow_down_factor_{slow_down_factor},
  head_start_{head_start},
  time_step_train_{time_step_train},
  is_analytic_env_{is_analytic_env},
  time_{(is_analytic_env) ? 0.0 : ros::Time::now().toSec()},
  initial_time_{time_},
  time_planner_{0.0} {};

double BaseGripperPlanner::updateTime(bool pause_gripper) {
  double dt;
  if (!is_analytic_env_) {
    dt = (ros::Time::now().toSec() - time_);
  } else {
    dt = time_step_train_;
  }

  time_ += dt;
  if (!pause_gripper) {
    time_planner_ += dt;
  }
  return dt;
}

bool BaseGripperPlanner::inStartPause() const {
  return (time_ - initial_time_) < head_start_;
}

bool BaseGripperPlanner::isDone(const tf::Transform &gripper_tf) const {
  // alternative: get a signal from the gripper trajectory planner that we are at the end
  double dist_to_goal = (gripper_goal_wrist_.getOrigin() - gripper_tf.getOrigin()).length();
  bool is_close = (dist_to_goal < success_thres_dist_);
  if (is_close) {
    // distance to target rotation: https://math.stackexchange.com/questions/90081/quaternion-distance
    double rot_distance = utils::calcRotDist(gripper_tf, gripper_goal_wrist_);
    // more exact alternative; seems to have some precision problems, returning nan if slightly above 1
    // double rot_distance = inner_prod > 1.0 ? 0.0 : acos(2.0 * pow(inner_prod, 2.0) - 1.0);
    is_close &= (rot_distance < success_thres_rot_);
  }
  return is_close;
}

EEObs BaseGripperPlanner::step(const RobotObs &robot_obs, const double &learned_vel_norm) {
  bool pause_gripper = inStartPause();
  double last_dt = updateTime(pause_gripper);
  GripperPlan gripper_plan;
  if (pause_gripper) {
    gripper_plan = getPrevPlan();
  } else {
    gripper_plan =
      internalStep(time_planner_ / slow_down_factor_, last_dt / slow_down_factor_, robot_obs, learned_vel_norm, !pause_gripper);
  }
  return gripper_plan.toEEObs(robot_obs, 0.0, false);
}

// to generate observations for the RL agent always use the training frequency and the unlearned velocity norms
EEObs BaseGripperPlanner::generateObsStep(const RobotObs &robot_obs) {
  GripperPlan gripper_plan =
    internalStep(time_planner_ / slow_down_factor_, inStartPause() ? 0.0 : time_step_train_, robot_obs, -1.0, false);
  bool done = isDone(utils::listToTf(robot_obs.gripper_tf));
  return gripper_plan.toEEObs(robot_obs, 0.0, done);
}
