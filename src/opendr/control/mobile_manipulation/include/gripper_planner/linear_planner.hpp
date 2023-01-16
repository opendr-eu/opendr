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

#ifndef MOBILE_MANIPULATION_RL_LINEAR_PLANNER_H
#define MOBILE_MANIPULATION_RL_LINEAR_PLANNER_H

#include "gripper_planner/base_gripper_planner.hpp"

class LinearPlanner : public BaseGripperPlanner {
private:
  double initial_dist_to_gripper_goal_;

  static tf::Vector3 getVel(const tf::Transform &current, const tf::Transform &goal, double min_vel, double max_vel);
  static tf::Quaternion getRot(const tf::Transform &initial, const tf::Transform &next, const tf::Transform &goal,
                               double initial_dist);

  GripperPlan calcNextStep(const GripperPlan &prev_plan, const double &dt, const double &min_velocity,
                           const double &max_velocity);

public:
  LinearPlanner(const std::vector<double> &gripper_goal_wrist, const std::vector<double> &initial_gripper_tf,
                const std::vector<double> &base_goal, const std::vector<double> &initial_base_tf, double success_thres_dist,
                double success_thres_rot, const double &min_planner_velocity, const double &max_planner_velocity,
                const double &slow_down_factor, const double &head_start, const double &time_step_train,
                const bool &is_analytic_env);

  GripperPlan internalStep(double time, double dt, const RobotObs &robot_obs, const double &learned_vel_norm,
                           bool update_prev_plan) override;
};

#endif  // MOBILE_MANIPULATION_RL_LINEAR_PLANNER_H