#include <gripper_planner/base_gripper_planner.h>

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
  // ROS_INFO("t-: %f, tp: %f, dt: %f, sp: %f", time_ - reset_time_, time_planner_, dt, start_pause_);
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

//
// PlannedVelocities BaseGripperPlanner::transformToVelocity(tf::Transform current,
//                                                          tf::Transform next,
//                                                          tf::Transform baseTransform,
//                                                          double upper_vel_limit){
//    PlannedVelocities planned_vel;
//
//    planned_vel.vel_world = next.getOrigin() - current.getOrigin();
//
//    if (upper_vel_limit != 0.0) {
//        // planned_vel.vel_world = utils::min_max_scale_vel(planned_vel.vel_world, 0.0, upper_vel_limit);
//        planned_vel.vel_world = utils::norm_scale_vel(planned_vel.vel_world, 0.0, upper_vel_limit);
//        // planned_vel.vel_world = utils::max_clip_vel(planned_vel.vel_world, upper_vel_limit);
//    }
//
//    tf::Transform base_no_trans = baseTransform;
//    base_no_trans.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
//    planned_vel.vel_rel = base_no_trans.inverse() * planned_vel.vel_world;
//
//    // planned change in rotation
//    // a) planned_vel.dq_world = next.getRotation() - current.getRotation();
//    // b) defined as dq * current == next
//    planned_vel.dq = (next.getRotation() * current.getRotation().inverse()).normalized();
//    // c) defined as dq * current == next in base frame
//    // tf::Transform current_rel = base_no_trans.inverse() * current;
//    // tf::Transform next_rel = base_no_trans.inverse() * next;
//    // planned_vel.dq_rel = (next_rel.getRotation() * current_rel.getRotation().inverse()).normalized();
//    // Note that these do not result in the same difference in RPY
//    // tf::Matrix3x3(planned_vel.dq_world).getRPY(roll_, pitch_, yaw_);
//    // tf::Matrix3x3(planned_vel.dq_rel).getRPY(roll2_, pitch2_, yaw2_);
//    // std::cout << "dq_world RPY roll: " << roll_ << ", pitch: " << pitch_ << ", yaw: " << yaw_ << std::endl;
//    // std::cout << "dq_rel RPY roll: " << roll2_ << ", pitch: " << pitch2_ << ", yaw: " << yaw2_ << std::endl;
//    // d) quaternion of the difference in RPY (identical in world and base frame. But different to all alternatives above)
//    // tf::Vector3 rpy_current = utils::q_to_rpy(current.getRotation());
//    // tf::Vector3 rpy_next = utils::q_to_rpy(next.getRotation());
//    // planned_vel.dq.setRPY(utils::rpy_angle_diff(rpy_next.x(), rpy_current.x()),
//    //                       utils::rpy_angle_diff(rpy_next.y(), rpy_current.y()),
//    //                       utils::rpy_angle_diff(rpy_next.z(), rpy_current.z()));
//    // planned_vel.dq.normalize();
//
//    return planned_vel;
//}
