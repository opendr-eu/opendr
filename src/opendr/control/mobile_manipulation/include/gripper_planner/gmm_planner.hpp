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

#ifndef MOBILE_MANIPULATION_RL_GMM_PLANNER_H
#define MOBILE_MANIPULATION_RL_GMM_PLANNER_H

#include "gripper_planner/base_gripper_planner.hpp"
#include "gripper_planner/gaussian_mixture_model.hpp"

class GMMPlanner : public BaseGripperPlanner {
private:
  boost::shared_ptr<GaussianMixtureModel> gaussian_mixture_model_;
  std::string gmm_model_path_;
  const tf::Vector3 tip_to_gripper_offset_;
  const tf::Quaternion gripper_to_base_rot_offset_;

  GripperPlan calcNextStep(double time, double dt, const tf::Vector3 &current_base_vel_world,
                           const tf::Vector3 &current_gripper_vel_world, const tf::Quaternion &current_gripper_dq,
                           const GripperPlan &prev_plan, const double &min_velocity, const double &max_velocity,
                           bool do_update);
  tf::Transform tipToObjOrigin(const tf::Transform &tip);
  GripperPlan getPrevPlan() override;

public:
  GMMPlanner(const std::vector<double> &gripper_goal_wrist, const std::vector<double> &initial_gripper_tf,
             const std::vector<double> &base_goal, const std::vector<double> &initial_base_tf, double success_thres_dist,
             double success_thres_rot, const double &min_planner_velocity, const double &max_planner_velocity,
             const double &slow_down_factor, const double &head_start, const double &time_step_train,
             const bool &is_analytic_env, const std::vector<double> &tip_to_gripper_offset,
             const std::vector<double> &gripper_to_base_rot_offset, std::string gmm_model_path, double gmm_base_offset);

  GripperPlan internalStep(double time, double dt, const RobotObs &robot_obs, const double &learned_vel_norm,
                           bool update_prev_plan) override;
  std::vector<tf::Transform> getMus();

  std::vector<double> objOriginToTip(const std::vector<double> &obj_origin);
};

#endif  // MOBILE_MANIPULATION_RL_GMM_PLANNER_H
