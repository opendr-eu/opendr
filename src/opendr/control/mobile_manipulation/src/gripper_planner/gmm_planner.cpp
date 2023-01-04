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

#include <gripper_planner/gmm_planner.hpp>

GMMPlanner::GMMPlanner(const std::vector<double> &gripper_goal_wrist, const std::vector<double> &initial_gripper_tf,
                       const std::vector<double> &base_goal, const std::vector<double> &initial_base_tf,
                       double success_thres_dist, double success_thres_rot, const double &min_planner_velocity,
                       const double &max_planner_velocity, const double &slow_down_factor, const double &head_start,
                       const double &time_step_train, const bool &is_analytic_env,
                       const std::vector<double> &tip_to_gripper_offset, const std::vector<double> &gripper_to_base_rot_offset,
                       std::string gmm_model_path, double gmm_base_offset) :
  BaseGripperPlanner(gripper_goal_wrist, initial_gripper_tf, base_goal, initial_base_tf, success_thres_dist, success_thres_rot,
                     min_planner_velocity, max_planner_velocity, slow_down_factor, head_start, time_step_train,
                     is_analytic_env),
  tip_to_gripper_offset_{utils::listToVector3(tip_to_gripper_offset)},
  gripper_to_base_rot_offset_{utils::listToQuaternion(gripper_to_base_rot_offset)},
  gmm_model_path_{gmm_model_path} {
  double max_rot = 0.1;
  gaussian_mixture_model_.reset(new GaussianMixtureModel(max_rot, max_rot));
  // For each learned object manipulation there are three action models one for grasping, one for manipulation and one for
  // releasing Stick to grasp and move of KallaxTuer first.
  gaussian_mixture_model_->loadFromFile(gmm_model_path_);
  // adaptModel(objectPose) transforms the GMM to a given object pose of the handled object
  // afterwards the model can be integrated step by step to generate new gripper poses leading to the correct handling of the
  // object
  tf::Transform gripper_goal_tip =
    utils::gripperToTipGoal(gripper_goal_wrist_, tip_to_gripper_offset_, gripper_to_base_rot_offset_);
  tf::Transform obj_origin_goal = tipToObjOrigin(gripper_goal_tip);
  gaussian_mixture_model_->adaptModel(obj_origin_goal, tf::Vector3(gmm_base_offset, 0, 0));

  // transform to a tip goal which is the frame the rest of the planner will work with
  prev_plan_.next_gripper_tf =
    utils::gripperToTipGoal(initial_gripper_tf_, tip_to_gripper_offset_, gripper_to_base_rot_offset_);
  prev_plan_.next_base_tf = initial_base_tf_;
};

tf::Transform GMMPlanner::tipToObjOrigin(const tf::Transform &tip) {
  tf::Transform related_object_pose = gaussian_mixture_model_->getRelatedObjPose();
  tf::Transform last_mu_gripper_orig = gaussian_mixture_model_->getLastMuEigenBckGripper();
  return tip * last_mu_gripper_orig.inverse() * related_object_pose;
}

std::vector<double> GMMPlanner::objOriginToTip(const std::vector<double> &obj_origin) {
  tf::Transform obj_origin_tf(utils::listToTf(obj_origin));
  tf::Transform related_object_pose = gaussian_mixture_model_->getRelatedObjPose();
  tf::Transform last_mu_gripper_orig = gaussian_mixture_model_->getLastMuEigenBckGripper();
  return utils::tfToList(obj_origin_tf * related_object_pose.inverse() * last_mu_gripper_orig);
}

GripperPlan GMMPlanner::calcNextStep(double time, double dt, const tf::Vector3 &current_base_vel_world,
                                     const tf::Vector3 &current_gripper_vel_world, const tf::Quaternion &current_gripper_dq,
                                     const GripperPlan &prev_plan, const double &min_velocity, const double &max_velocity,
                                     bool do_update) {
  // create eigen vectors for current pose and current speed
  // treat it as a planner that could be pre-computed: calculate next step from transform we wanted to achieve, not the actually
  // achieved
  Eigen::VectorXf current_pose(14);
  current_pose << prev_plan.next_gripper_tf.getOrigin().x(), prev_plan.next_gripper_tf.getOrigin().y(),
    prev_plan.next_gripper_tf.getOrigin().z(), prev_plan.next_gripper_tf.getRotation().x(),
    prev_plan.next_gripper_tf.getRotation().y(), prev_plan.next_gripper_tf.getRotation().z(),
    prev_plan.next_gripper_tf.getRotation().w(), prev_plan.next_base_tf.getOrigin().x(), prev_plan.next_base_tf.getOrigin().y(),
    0.0, prev_plan.next_base_tf.getRotation().x(), prev_plan.next_base_tf.getRotation().y(),
    prev_plan.next_base_tf.getRotation().z(), prev_plan.next_base_tf.getRotation().w();

  // use the planned velocities as current velocities, assuming we follow a pre-calculated plan as above
  Eigen::VectorXf current_speed(14);
  current_speed << current_gripper_vel_world.x(), current_gripper_vel_world.y(), current_gripper_vel_world.z(),
    current_gripper_dq.x(), current_gripper_dq.y(), current_gripper_dq.z(), current_gripper_dq.w(), current_base_vel_world.x(),
    current_base_vel_world.y(), 0.0, 0.0, 0.0, 0.0, 0.0;

  gaussian_mixture_model_->integrateModel(time, dt, &current_pose, &current_speed, min_velocity, max_velocity, do_update);

  GripperPlan nextPlan;
  nextPlan.next_gripper_tf.setOrigin(tf::Vector3(current_pose[0], current_pose[1], current_pose[2]));
  nextPlan.next_gripper_tf.setRotation(tf::Quaternion(current_pose[3], current_pose[4], current_pose[5], current_pose[6]));
  nextPlan.next_base_tf.setOrigin(tf::Vector3(current_pose[7], current_pose[8], 0.0));
  nextPlan.next_base_tf.setRotation(tf::Quaternion(current_pose[10], current_pose[11], current_pose[12], current_pose[13]));
  return nextPlan;
}

GripperPlan GMMPlanner::getPrevPlan() {
  GripperPlan prev_plan_wrist = prev_plan_;
  prev_plan_wrist.next_gripper_tf =
    utils::tipToGripperGoal(prev_plan_wrist.next_gripper_tf, tip_to_gripper_offset_, gripper_to_base_rot_offset_);
  return prev_plan_wrist;
}

GripperPlan GMMPlanner::internalStep(double time, double dt, const RobotObs &robot_obs, const double &learned_vel_norm,
                                     bool update_prev_plan) {
  double min_vel, max_vel;
  if (learned_vel_norm >= 0.0) {
    min_vel = learned_vel_norm;
    max_vel = learned_vel_norm;
  } else {
    min_vel = 0.0;
    max_vel = max_planner_velocity_;
  }

  tf::Transform gripper_velocities = utils::listToTf(robot_obs.gripper_velocities);
  GripperPlan next_plan = calcNextStep(time, dt, utils::listToVector3(robot_obs.base_velocity), gripper_velocities.getOrigin(),
                                       gripper_velocities.getRotation(), prev_plan_, min_vel, max_vel, update_prev_plan);

  if (update_prev_plan) {
    prev_plan_ = next_plan;
  }

  GripperPlan next_plan_wrist = next_plan;
  next_plan_wrist.next_gripper_tf =
    utils::tipToGripperGoal(next_plan.next_gripper_tf, tip_to_gripper_offset_, gripper_to_base_rot_offset_);

  return next_plan_wrist;
}

std::vector<tf::Transform> GMMPlanner::getMus() {
  int nrModes = gaussian_mixture_model_->getNrModes();
  std::vector<tf::Transform> v;

  for (int i = 0; i < nrModes; i++) {
    std::vector<Eigen::VectorXf> muEigen = gaussian_mixture_model_->getMu();

    tf::Transform gripper_t;
    gripper_t.setOrigin(tf::Vector3(muEigen[i][1], muEigen[i][2], muEigen[i][3]));
    gripper_t.setRotation(tf::Quaternion(muEigen[i][4], muEigen[i][5], muEigen[i][6], muEigen[i][7]));

    tf::Transform base_t;
    base_t.setOrigin(tf::Vector3(muEigen[i][8], muEigen[i][9], 0.0));
    base_t.setRotation(tf::Quaternion(muEigen[i][11], muEigen[i][12], muEigen[i][13], muEigen[i][14]));

    v.push_back(gripper_t);
    v.push_back(base_t);
  }
  return v;
}