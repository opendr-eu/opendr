#pragma once

#include "gripper_planner/base_gripper_planner.h"

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