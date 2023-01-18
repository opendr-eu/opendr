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

#include "mobile_manipulation_rl/robot_env.hpp"

RobotEnv::RobotEnv(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
                   double time_step_world, bool perform_collision_check, std::string node_handle_name, bool verbose,
                   std::string robo_conf_path) :
  // ros::init will simply ignore new calls to it if creating multiple envs I believe
  ROSCommonNode(0, NULL, "robot_env"),
  nh_{new ros::NodeHandle(node_handle_name)},
  rate_{1.0 / time_step_world},
  rng_{seed},
  verbose_{verbose},
  robo_conf_path_{robo_conf_path},
  robo_config_{RoboConf::fromYaml(robo_conf_path)},
  strategy_{strategy},
  init_controllers_{init_controllers},
  penalty_scaling_{penalty_scaling},
  time_step_train_{0.1},
  perform_collision_check_{perform_collision_check} {
  gripper_visualizer_ = nh_->advertise<visualization_msgs::Marker>("gripper_goal_visualizer", 1);
  robstate_visualizer_ = nh_->advertise<moveit_msgs::DisplayRobotState>("robot_state_visualizer", 50);
  base_cmd_pub_ = nh_->advertise<geometry_msgs::Twist>(robo_config_.base_cmd_topic, 1);
  client_get_scene_ = nh_->serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");

  // Load Robot config from moveit movegroup (must be running)
  robot_model_loader::RobotModelLoaderPtr robot_model_loader;
  robot_model_loader.reset(new robot_model_loader::RobotModelLoader("robot_description"));

  robot_model::RobotModelPtr kinematic_model = robot_model_loader->getModel();
  kinematic_state_.reset(new robot_state::RobotState(kinematic_model));
  kinematic_state_->setToDefaultValues();

  joint_model_group_ = kinematic_model->getJointModelGroup(robo_config_.joint_model_group_name);
  joint_names_ = joint_model_group_->getVariableNames();

  planning_scene_.reset(new planning_scene::PlanningScene(kinematic_model));
  ROS_INFO("Planning frame: %s", planning_scene_->getPlanningFrame().c_str());

  moveit_msgs::GetPlanningScene scene_srv1;
  scene_srv1.request.components.components = 2;  // moveit_msgs::PlanningSceneComponents::ROBOT_STATE;
  if (!client_get_scene_.call(scene_srv1)) {
    ROS_WARN("Failed to call service /get_planning_scene");
  }
  planning_scene_->setPlanningSceneDiffMsg(scene_srv1.response.scene);
  robot_state::RobotState robstate = planning_scene_->getCurrentState();

  // moveit initialises all joints to 0. If we use gazebo or similar, first update all values to match the simulator
  if (init_controllers_) {
    const std::vector<std::string> &all_joint_names = kinematic_model->getJointModelNames();
    for (const auto &name : all_joint_names) {
      const double default_value = kinematic_state_->getJointPositions(name)[0];
      const double actual_value = robstate.getJointPositions(name)[0];
      // avoid adding joints that are not defined in other places (e.g. rviz)
      if (std::abs(default_value - actual_value) > 0.0000001) {
        // update the values in the kinematic_state which are simply set to default above
        kinematic_state_->setJointPositions(name, &actual_value);
        ROS_INFO("name: %s, default: %f, actual: %f", name.c_str(), default_value, actual_value);
      }
    }
    // in the analytical env set non-default joints to the hardcoded values that we know we will use in the simulator later
  } else {
    for (auto const &x : robo_config_.initial_joint_values) {
      kinematic_state_->setJointPositions(x.first, &x.second);
    }
  }

  setWorld(world_type);

  // always do this so we can later change to real_execution
  if (init_controllers_) {
    spinner_ = new ros::AsyncSpinner(2);
    spinner_->start();
  }

  if (perform_collision_check_) {
    // Collision constraint function GroupStateValidityCallbackFn(),
    moveit_msgs::GetPlanningScene scene_srv;
    moveit_msgs::PlanningScene currentScene;
    scene_srv.request.components.components = 24;  // moveit_msgs::PlanningSceneComponents::WORLD_OBJECT_NAMES;
    if (!client_get_scene_.call(scene_srv)) {
      ROS_WARN("Failed to call service /get_planning_scene");
    }
    currentScene = scene_srv.response.scene;
    ROS_INFO("Known collision objects:");
    for (auto &collision_object : scene_srv.response.scene.world.collision_objects) {
      ROS_INFO_STREAM(collision_object.id);
    }
    planning_scene_->setPlanningSceneDiffMsg(currentScene);
    constraint_callback_fn_ = boost::bind(&validity_fun::validityCallbackFn, planning_scene_, kinematic_state_, _2, _3);
  }
}

void RobotEnv::setWorld(const std::string &world_type) {
  if (world_type == "gazebo") {
    world_ = new GazeboWorld();
  } else if (world_type == "world") {
    world_ = new RealWorld();
  } else if (world_type == "sim") {
    world_ = new SimWorld();
  } else {
    throw std::runtime_error("Unknown real_execution value");
  }
  if ((!world_->isAnalytical()) && !init_controllers_) {
    throw std::runtime_error("must have initialised controllers to use real_execution_");
  }
}

bool RobotEnv::setInitialGripperPose(const std::string &initial_joint_distribution) {
  if (initial_joint_distribution == "fixed") {
    kinematic_state_->setVariablePositions(robo_config_.neutral_pos_joint_names, robo_config_.neutral_pos_values);
  } else if ((initial_joint_distribution == "rnd") || (initial_joint_distribution == "restricted_ws")) {
    collision_detection::CollisionRequest collision_request;
    collision_request.group_name = robo_config_.joint_model_group_name;
    collision_detection::CollisionResult collision_result;

    bool invalid = true;
    while (invalid) {
      kinematic_state_->setToRandomPositions(joint_model_group_, rng_);

      // check if in self-collision
      planning_scene_->getCurrentStateNonConst().update();

      robot_state::RobotState state_copy(*kinematic_state_);
      state_copy.setVariablePosition("world_joint/x", robot_state_.base_tf.getOrigin().x());
      state_copy.setVariablePosition("world_joint/y", robot_state_.base_tf.getOrigin().y());
      state_copy.setVariablePosition("world_joint/theta", robot_state_.base_tf.getRotation().getAngle() *
                                                            robot_state_.base_tf.getRotation().getAxis().getZ());

      planning_scene_->checkCollisionUnpadded(collision_request, collision_result, state_copy);
      invalid = collision_result.collision;
      ROS_INFO_COND(collision_result.collision, "set_start_pose: drawn pose in self-collision, trying again");
      collision_result.clear();

      if (initial_joint_distribution == "restricted_ws") {
        const Eigen::Affine3d &ee_pose = kinematic_state_->getGlobalLinkTransform(robo_config_.global_link_transform);
        tf::Transform temp_tf;
        tf::transformEigenToTF(ee_pose, temp_tf);
        invalid &= (temp_tf.getOrigin().z() < robo_config_.restricted_ws_z_min) ||
                   (temp_tf.getOrigin().z() > robo_config_.restricted_ws_z_max);
        ROS_INFO_COND(invalid, "Goal outside of restricted ws, sampling again.");
      }
    }
  } else {
    throw std::runtime_error("Invalid start_pose_distribution");
  }

  const Eigen::Affine3d &end_effector_state = kinematic_state_->getGlobalLinkTransform(robo_config_.global_link_transform);
  tf::transformEigenToTF(end_effector_state, robot_state_.relative_gripper_tf);
  robot_state_.gripper_tf = robot_state_.base_tf * robot_state_.relative_gripper_tf;
  kinematic_state_->copyJointGroupPositions(joint_model_group_, robot_state_.joint_values);

  // arm: use controllers
  bool success = true;
  if (!world_->isAnalytical()) {
    ROS_INFO("Setting gripper to start");
    sendArmCommand(robot_state_.joint_values, 5.0);
    success = getArmSuccess();
    ROS_WARN_COND(!success, "couldn't set arm to selected start pose");
  }
  return success;
}

void RobotEnv::setInitialPose(const std::vector<double> &initial_base_pose, const std::string &initial_joint_distribution) {
  // set base
  tf::Transform initial_base_tf = utils::listToTf(initial_base_pose);
  if (world_->getName() == "world") {
    ROS_INFO("Real world execution set. Taking the current base transform as starting point.");
    robot_state_.base_tf = world_->getBaseTransformWorld();
    if (utils::tfAlmostEqual(initial_base_tf, robot_state_.base_tf)) {
      throw std::runtime_error("Initial base tf not matching");
    }
  } else {
    robot_state_.base_tf = utils::listToTf(initial_base_pose);
  }
  world_->setModelState(robo_config_.name, robot_state_.base_tf, robo_config_, base_cmd_pub_);

  // set gripper: if not the analytical env, we actually execute it in gazebo to reset.
  // This might sometimes fail. So continue sampling a few random poses
  bool success = false;
  int trials = 0, max_trials = 50;
  while ((!success) && trials < max_trials) {
    success = setInitialGripperPose(initial_joint_distribution);
    trials++;
  }
  if (trials > max_trials) {
    throw std::runtime_error("Could not set start pose after 50 trials!!!");
  }
}

RobotObs RobotEnv::reset(const std::vector<double> &initial_base_pose, const std::string &initial_joint_distribution,
                         bool do_close_gripper, double success_thres_dist, double success_thres_rot) {
  ROS_INFO_COND(!world_->isAnalytical(), "Reseting environment");

  success_thres_dist_ = success_thres_dist;
  success_thres_rot_ = success_thres_rot;
  setInitialPose(initial_base_pose, initial_joint_distribution);

  if (init_controllers_) {
    do_close_gripper ? closeGripper(0.0, false) : openGripper(0.08, false);
  }

  // reset time after the start pose is set
  time_ = (world_->isAnalytical()) ? 0.0 : ros::Time::now().toSec();

  // Clear the visualizations
  if (verbose_) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = robo_config_.frame_id;
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::DELETEALL;
    gripper_visualizer_.publish(marker);

    gripper_plan_marker_.markers.clear();
    marker_counter_++;
    if (marker_counter_ > 3) {
      marker_counter_ = 0;
    }
  }
  pathPoints_.clear();
  addTrajectoryPoint(robot_state_.gripper_tf, true);

  prev_robot_state_ = robot_state_;

  return robot_state_.toRobotObs(prev_robot_state_, false, 0.0, false);
}

tf::Vector3 RobotEnv::worldToRelativeEEVelocities(const tf::Vector3 &ee_vel_world) const {
  tf::Transform base_no_trans = robot_state_.base_tf;
  base_no_trans.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
  return base_no_trans.inverse() * ee_vel_world;
}

RobotObs RobotEnv::step(std::vector<double> base_actions, const std::vector<double> &ee_velocities_world,
                        double transition_noise_base) {
  PathPoint path_point;
  prev_robot_state_ = robot_state_;

  tf::Transform ee_vel_world = utils::listToTf(ee_velocities_world);
  tf::Vector3 planned_gripper_vel_rel = worldToRelativeEEVelocities(ee_vel_world.getOrigin());

  tf::Transform desired_base_tf;
  bool collision = false;
  double regularization = 0.0;
  double last_dt = updateTime();

  geometry_msgs::Twist base_cmd_rel =
    calcDesiredBaseTransform(base_actions, utils::normScaleVel(planned_gripper_vel_rel, 0.0, robo_config_.base_vel_rng),
                             desired_base_tf, transition_noise_base, regularization, last_dt);

  tf::Transform desired_gripper_tf;
  desired_gripper_tf.setOrigin(robot_state_.gripper_tf.getOrigin() + ee_vel_world.getOrigin());
  desired_gripper_tf.setRotation(ee_vel_world.getRotation() * robot_state_.gripper_tf.getRotation());

  tf::Transform desired_gripper_tf_rel = desired_base_tf.inverse() * desired_gripper_tf;

  // Perform IK checks
  Eigen::Isometry3d state;
  tf::poseTFToEigen(desired_gripper_tf_rel, state);
  const Eigen::Isometry3d &desired_gripper_eigen_rel = state;
  bool found_ik = findIk(desired_gripper_eigen_rel, desired_gripper_tf);
  kinematic_state_->copyJointGroupPositions(joint_model_group_, robot_state_.joint_values);

  if ((!world_->isAnalytical())) {
    rate_.sleep();
    sendArmCommand(robot_state_.joint_values, 0.1);
    base_cmd_pub_.publish(base_cmd_rel);
  }

  // update state to what we actually achieve
  // a) base: without execution we'll always be at the next base transform
  if (world_->isAnalytical()) {
    robot_state_.base_tf = desired_base_tf;
  } else {
    robot_state_.base_tf = world_->getBaseTransformWorld();
  }
  // b) gripper: update kinematic state from planning scene and run forward kinematics to get achieved currentGripperTransform_
  const Eigen::Affine3d &end_effector_state_rel = kinematic_state_->getGlobalLinkTransform(robo_config_.global_link_transform);
  tf::transformEigenToTF(end_effector_state_rel, robot_state_.relative_gripper_tf);
  robot_state_.gripper_tf = robot_state_.base_tf * robot_state_.relative_gripper_tf;
  // update_current_gripper_from_world();

  // there seems to be an incompatibility with some geometries leading to occasional segfaults within
  // planning_scene::PlanningScene::checkCollisionUnpadded
  // if (init_controllers_){
  //     collision |= check_scene_collisions();
  // }

  addTrajectoryPoint(desired_gripper_tf, found_ik);

  // reward and check if episode has finished -> Distance gripper to goal
  // found_ik &= get_arm_success();
  double reward = calcReward(found_ik, regularization);
  bool done = found_ik;

  pathPoints_.push_back(path_point);

  return robot_state_.toRobotObs(prev_robot_state_, !found_ik, reward, done);
}

// NOTE: the other parts of the reward (action regularization) happens in python
double RobotEnv::calcReward(bool found_ik, double regularization) {
  double reward = -penalty_scaling_ * regularization;
  if (!found_ik) {
    reward -= 1.0;
  }
  return reward;
}

bool RobotEnv::findIk(const Eigen::Isometry3d &desired_state_eigen, const tf::Transform &desiredGripperTfWorld) {
  // kinematics::KinematicsQueryOptions ik_options;
  // ik_options.return_approximate_solution = true;
  if (perform_collision_check_) {
    bool success = kinematic_state_->setFromIK(joint_model_group_, desired_state_eigen, robo_config_.kinematics_solver_timeout,
                                               constraint_callback_fn_);
    if (!success) {
      // in case of a collision keep the current position
      // can apply this to any case of ik failure as moveit does not seem to set it to the next best solution anyway
      kinematic_state_->setJointGroupPositions(robo_config_.joint_model_group_name, robot_state_.joint_values);
    }
    return success;
  } else {
    // return kinematic_state_->setFromIK(joint_model_group_, desiredState, 5, 0.1,
    // moveit::core::GroupStateValidityCallbackFn(), ik_options);
    return kinematic_state_->setFromIK(joint_model_group_, desired_state_eigen, robo_config_.kinematics_solver_timeout);
  }
}

double RobotEnv::updateTime() {
  double dt;
  if ((!world_->isAnalytical())) {
    dt = (ros::Time::now().toSec() - time_);
  } else {
    dt = time_step_train_;
  }
  time_ += dt;
  return dt;
}

void RobotEnv::addTrajectoryPoint(const tf::Transform &desired_gripper_tf, bool found_ik) {
  if (!verbose_) {
    return;
  }
  // plans
  double nthpoint = (world_->isAnalytical()) ? (1.0 / time_step_train_) : (1.0 / (0.02));
  if (((pathPoints_.size() % (int)nthpoint) == 0) || !found_ik) {
    int mid = marker_counter_ + gripper_plan_marker_.markers.size();
    std::string ik_color = found_ik ? "green" : "red";
    visualization_msgs::Marker marker = utils::markerFromTransform(
      desired_gripper_tf, "gripper_plan", utils::getColorMsg(ik_color, 0.5), mid, robo_config_.frame_id);
    gripper_visualizer_.publish(marker);
    gripper_plan_marker_.markers.push_back(marker);

    visualization_msgs::Marker base_marker = utils::markerFromTransform(
      robot_state_.base_tf, "base_actual", utils::getColorMsg("yellow", 0.5), mid, robo_config_.frame_id);
    gripper_visualizer_.publish(base_marker);
  };
  // current robot state
  if ((pathPoints_.size() % (int)nthpoint) == 0) {
    moveit_msgs::DisplayRobotState drs;
    robot_state::RobotState state_copy(*kinematic_state_);
    state_copy.setVariablePosition("world_joint/x", robot_state_.base_tf.getOrigin().x());
    state_copy.setVariablePosition("world_joint/y", robot_state_.base_tf.getOrigin().y());
    state_copy.setVariablePosition(
      "world_joint/theta", robot_state_.base_tf.getRotation().getAngle() * robot_state_.base_tf.getRotation().getAxis().getZ());
    robot_state::robotStateToRobotStateMsg(state_copy, drs.state);
    robstate_visualizer_.publish(drs);
  }
}

void RobotEnv::publishMarker(const std::vector<double> &marker_tf, int marker_id, const std::string &name_space,
                             const std::string &color, double alpha, const std::string &geometry,
                             const std::vector<double> &marker_scale) {
  tf::Transform t = utils::listToTf(marker_tf);
  tf::Vector3 scale = utils::listToVector3(marker_scale);
  std_msgs::ColorRGBA c = utils::getColorMsg(color, alpha);
  visualization_msgs::Marker marker =
    utils::markerFromTransform(t, name_space, c, marker_id, robo_config_.frame_id, geometry, scale);
  gripper_visualizer_.publish(marker);
}

std::vector<double> RobotEnv::tipToGripperTf(const std::vector<double> &tip_tf) {
  tf::Transform tf = utils::listToTf(tip_tf);
  return utils::tfToList(
    utils::tipToGripperGoal(tf, robo_config_.tip_to_gripper_offset, robo_config_.gripper_to_base_rot_offset));
}

std::vector<double> RobotEnv::gripperToTipTf(const std::vector<double> &tip_tf) {
  tf::Transform tf = utils::listToTf(tip_tf);
  return utils::tfToList(
    utils::gripperToTipGoal(tf, robo_config_.tip_to_gripper_offset, robo_config_.gripper_to_base_rot_offset));
}

geometry_msgs::Twist RobotOmniDrive::calcDesiredBaseTransform(std::vector<double> &base_actions,
                                                              tf::Vector3 planned_gripper_vel_rel,
                                                              tf::Transform &desired_base_tf, double transition_noise_base,
                                                              double &regularization, double const &last_dt) {
  // a) calculate the new desire baseTransform
  // planner actions are based on last_dt, RL actions are for a unit time -> scale down RL actions
  double base_rot_rng_t = last_dt * robo_config_.base_rot_rng;
  double base_vel_rng_t = last_dt * robo_config_.base_vel_rng;

  // Modulate planned base velocity and set it:
  // i) derive actions from agent's actions
  tf::Vector3 base_vel_rel;
  double base_rotation = 0.0;

  if ((strategy_ == "relvelm") || (strategy_ == "relveld")) {
    // ALTERNATIVE WOULD BE TO INTERPRET modulation_lambda1 AS VELOCITY AND modulation_lambda2 AS ANGLE, THEN MOVE VEL * COS(X)
    // AND VEL * SIN(X). See Tiago.
    double dx = base_vel_rng_t * base_actions[1];
    double dy = base_vel_rng_t * base_actions[2];
    base_vel_rel.setValue(planned_gripper_vel_rel.x() + dx, planned_gripper_vel_rel.y() + dy, 0.0);
    base_rotation = base_rot_rng_t * base_actions[0];

    if (strategy_ == "relvelm") {
      // a) modulate as little as possible
      regularization += pow(base_actions[0], 2.0) + pow(base_actions[1], 2.0) + pow(base_actions[2], 2.0);
    } else {
      // b) keep total speed low (scaled back up into -1, 1 range)
      double denom = std::abs(base_vel_rng_t) < 0.000001 ? 1.0 : base_vel_rng_t;
      regularization += pow(base_actions[0], 2.0) + pow(base_vel_rel.length() / denom, 2.0);
    }
  } else if (strategy_ == "dirvel") {
    double dx = base_vel_rng_t * base_actions[1];
    double dy = base_vel_rng_t * base_actions[2];
    base_vel_rel.setValue(dx, dy, 0.0);
    base_rotation = base_rot_rng_t * base_actions[0];

    regularization += pow(base_actions[0], 2.0) + pow(base_actions[1], 2.0) + pow(base_actions[2], 2.0);
  } else {
    throw std::runtime_error("Unimplemented strategy");
  }

  // ensure the velocity limits are still satisfied
  base_vel_rel = utils::normScaleVel(base_vel_rel, 0.0, base_vel_rng_t);
  // ensure z component is 0 (relevant for 'hack' in unmodulated strategy)
  base_vel_rel.setZ(0.0);

  if (transition_noise_base > 0.0001) {
    tf::Vector3 noise_vec =
      tf::Vector3(rng_.gaussian(0.0, transition_noise_base), rng_.gaussian(0.0, transition_noise_base), 0.0);
    base_vel_rel += noise_vec;
    base_rotation += rng_.gaussian(0.0, transition_noise_base);
  }

  // ii) set corresponding new base speed
  desired_base_tf = robot_state_.base_tf;

  tf::Transform base_no_trans = robot_state_.base_tf;
  base_no_trans.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
  // from robot-base reference frame back to global reference frame
  tf::Vector3 base_vel = base_no_trans * base_vel_rel;
  desired_base_tf.setOrigin(robot_state_.base_tf.getOrigin() + base_vel);

  // iii) rotate base
  tf::Quaternion q(tf::Vector3(0.0, 0.0, 1.0), base_rotation);
  desired_base_tf.setRotation(q * robot_state_.base_tf.getRotation());

  // construct base command: scale back up to be per unit time
  double cmd_scaling = 1.0;
  if (last_dt > 0.001) {
    cmd_scaling /= last_dt;
  }

  tf::Transform relative_desired_pose = robot_state_.base_tf.inverse() * desired_base_tf;
  geometry_msgs::Twist base_cmd_rel;
  double roll_, pitch_, yaw;
  relative_desired_pose.getBasis().getRPY(roll_, pitch_, yaw);
  base_cmd_rel.linear.x = relative_desired_pose.getOrigin().getX() * cmd_scaling;
  base_cmd_rel.linear.y = relative_desired_pose.getOrigin().getY() * cmd_scaling;
  base_cmd_rel.angular.z = yaw * cmd_scaling;

  return base_cmd_rel;
}

geometry_msgs::Twist RobotDiffDrive::calcDesiredBaseTransform(std::vector<double> &base_actions,
                                                              tf::Vector3 planned_gripper_vel_rel_rel,
                                                              tf::Transform &desired_base_tf, double transition_noise_base,
                                                              double &regularization, double const &last_dt) {
  // a) calculate the new desire baseTransform
  // planner actions are based on last_dt, RL actions are for a unit time -> scale down RL actions
  double base_rot_rng_t = last_dt * robo_config_.base_rot_rng;
  double base_vel_rng_t = last_dt * robo_config_.base_vel_rng;
  double vel_forward, angle;

  if (strategy_ == "dirvel") {
    vel_forward = base_vel_rng_t * base_actions[0];
    angle = base_rot_rng_t * base_actions[1];
    regularization += pow(base_actions[0], 2.0) + pow(base_actions[1], 2.0);
  } else {
    // velocity we want to achieve, then apply the motion constraints to it
    tf::Vector3 goal_vel_vec;
    double dvel;
    double dangle;
    if ((strategy_ == "relvelm") || (strategy_ == "relveld")) {
      dvel = base_vel_rng_t * base_actions[0];
      dangle = base_rot_rng_t * base_actions[1];
      goal_vel_vec = tf::Vector3(planned_gripper_vel_rel_rel.x(), planned_gripper_vel_rel_rel.y(), 0.0);
    } else {
      throw std::runtime_error("Unimplemented strategy for tiago");
    }

    vel_forward = goal_vel_vec.length();
    if (goal_vel_vec.x() < 0.0) {
      vel_forward = -vel_forward;
    }
    vel_forward += dvel;

    // angle to rotate by as angle between just moving forward and the gripper movement
    tf::Vector3 vel_forward_vec = tf::Vector3(vel_forward, 0.0, 0.0);
    angle =
      atan2(vel_forward_vec.x() * goal_vel_vec.y() - vel_forward_vec.y() * goal_vel_vec.x(), vel_forward_vec.dot(goal_vel_vec));
    angle = utils::clampDouble(angle, -base_rot_rng_t, base_rot_rng_t);
    angle += dangle;

    // enforce constraints
    vel_forward = utils::clampDouble(vel_forward, -base_vel_rng_t, base_vel_rng_t);
    angle = utils::clampDouble(angle, -base_rot_rng_t, base_rot_rng_t);

    if (strategy_ == "relvelm") {
      // a) modulate as little as possible
      regularization += pow(base_actions[0], 2.0) + pow(base_actions[1], 2.0);
    } else {
      // b) keep total speed low (scaled back up into -1, 1 range)
      regularization += pow(base_actions[0], 1) + pow(vel_forward / base_vel_rng_t, 2);
    }
  }

  if (transition_noise_base > 0.0001) {
    vel_forward += rng_.gaussian(0.0, transition_noise_base);
    angle += rng_.gaussian(0.0, transition_noise_base);
  }

  // calculate base transform after applying the new velocities
  desired_base_tf = robot_state_.base_tf;
  tf::Quaternion q(tf::Vector3(0.0, 0.0, 1.0), angle);
  desired_base_tf.setRotation(q * robot_state_.base_tf.getRotation());

  // world reference frame: move relative to base_angle defined by now rotated base
  double base_angle = desired_base_tf.getRotation().getAngle() * desired_base_tf.getRotation().getAxis().getZ();
  tf::Vector3 base_velocity_dif_drive(vel_forward * cos(base_angle), vel_forward * sin(base_angle), 0.0);
  desired_base_tf.setOrigin(robot_state_.base_tf.getOrigin() + base_velocity_dif_drive);

  // calculate corresponding cmd, scaled up to a unit time again
  double cmd_scaling = 1.0;
  if (last_dt > 0.001) {
    cmd_scaling /= last_dt;
  }
  geometry_msgs::Twist base_cmd_rel;
  base_cmd_rel.linear.x = vel_forward * cmd_scaling;
  base_cmd_rel.angular.z = angle * cmd_scaling;

  return base_cmd_rel;
}

// Callback for collision checking in ik search
namespace validity_fun {
  bool validityCallbackFn(planning_scene::PlanningScenePtr &planning_scene, const robot_state::RobotStatePtr &kinematic_state,
                          const robot_state::JointModelGroup *joint_model_group, const double *joint_group_variable_values) {
    kinematic_state->setJointGroupPositions(joint_model_group, joint_group_variable_values);
    // Now check for collisions
    collision_detection::CollisionRequest collision_request;
    collision_request.group_name = joint_model_group->getName();
    collision_detection::CollisionResult collision_result;
    // collision_detection::AllowedCollisionMatrix acm = planning_scene->getAllowedCollisionMatrix();
    planning_scene->getCurrentStateNonConst().update();
    planning_scene->checkCollisionUnpadded(collision_request, collision_result, *kinematic_state);
    // planning_scene->checkSelfCollision(collision_request, collision_result, *kinematic_state);

    if (collision_result.collision) {
      // ROS_INFO("IK solution is in collision!");
      return false;
    }
    return true;
  }
}  // namespace validity_fun