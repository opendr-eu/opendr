#ifndef MOBILE_MANIPULATION_RL_BASE_GRIPPER_PLANNER_H
#define MOBILE_MANIPULATION_RL_BASE_GRIPPER_PLANNER_H

#include <eigen_conversions/eigen_msg.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "std_msgs/ColorRGBA.h"
#include "tf/transform_datatypes.h"
#include "visualization_msgs/MarkerArray.h"

#include "../mobile_manipulation_rl/utils.h"

struct EEObs {
  const std::vector<double> next_gripper_tf;
  const std::vector<double> next_base_tf;
  const std::vector<double> ee_velocities_world;
  const std::vector<double> ee_velocities_rel;
  const double reward;
  const bool done;
};

struct GripperPlan {
  tf::Transform next_gripper_tf;
  tf::Transform next_base_tf;

  EEObs toEEObs(const RobotObs &robot_obs, const double &reward, const bool &done) {
    tf::Transform current_gripper_tf(utils::listToTf(robot_obs.gripper_tf));
    tf::Vector3 vel_world(next_gripper_tf.getOrigin() - current_gripper_tf.getOrigin());

    tf::Transform base_no_trans(utils::listToTf(robot_obs.base_tf));
    base_no_trans.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
    tf::Vector3 vel_rel(base_no_trans.inverse() * vel_world);

    tf::Quaternion dq = utils::calcDq(current_gripper_tf.getRotation(), next_gripper_tf.getRotation());
    return EEObs{utils::tfToList(next_gripper_tf, true),
                 utils::tfToList(next_base_tf, true),
                 utils::tfToList(tf::Transform(dq, vel_world), true),
                 utils::tfToList(tf::Transform(dq, vel_rel), true),
                 reward,
                 done};
  }
};

class BaseGripperPlanner {
private:
  const double slow_down_factor_;
  double time_;
  double initial_time_;
  double time_planner_;
  const double head_start_;
  const double time_step_train_;
  const bool is_analytic_env_;
  bool inStartPause() const;
  bool isDone(const tf::Transform &gripper_tf) const;
  double updateTime(bool pause_gripper);

protected:
  const tf::Transform gripper_goal_wrist_;
  const tf::Transform base_goal_;
  const tf::Transform initial_gripper_tf_;
  const tf::Transform initial_base_tf_;
  const double success_thres_dist_;
  const double success_thres_rot_;
  const double min_planner_velocity_;
  const double max_planner_velocity_;

  GripperPlan prev_plan_;
  virtual GripperPlan getPrevPlan() { return prev_plan_; };

  virtual GripperPlan internalStep(double time, double dt, const RobotObs &robot_obs, const double &learned_vel_norm,
                                   bool update_prev_plan) = 0;

public:
  BaseGripperPlanner(const std::vector<double> &gripper_goal_wrist, const std::vector<double> &initial_gripper_tf,
                     const std::vector<double> &base_goal, const std::vector<double> &initial_base_tf,
                     double success_thres_dist, double success_thres_rot, const double &min_planner_velocity,
                     const double &max_planner_velocity, const double &slow_down_factor, const double &head_start,
                     const double &time_step_train, const bool &is_analytic_env);

  EEObs step(const RobotObs &robot_obs, const double &learned_vel_norm);

  EEObs generateObsStep(const RobotObs &robot_obs);
};

  #endif  // MOBILE_MANIPULATION_RL_BASE_GRIPPER_PLANNER_H
