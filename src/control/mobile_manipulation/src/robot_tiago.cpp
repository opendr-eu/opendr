#include <modulation_rl/robot_tiago.h>

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

    // move_group_arm_torso_ = new moveit::planning_interface::MoveGroupInterface(joint_model_group_name_);
    // move_group_arm_torso_->setPlannerId("SBLkConfigDefault");
    // move_group_arm_torso_->setMaxVelocityScalingFactor(1.0);
    // //move_group_arm_torso_->setMaxAccelerationScalingFactor(0.05);

    gripper_client_.reset(new TrajClientTiago("/gripper_controller/follow_joint_trajectory"));
    while (!gripper_client_->waitForServer(ros::Duration(5.0))) {
      ROS_INFO("Waiting for the gripper_controller/follow_joint_trajectory action server to come up");
    }

    // somehow the controller manager expects the hash from a pr2_mechanism_msgs::SwitchController, not
    // controller_manager_msgs::SwitchController switch_controller_client_ =
    // nh_->serviceClient<pr2_mechanism_msgs::SwitchController>("/controller_manager/switch_controller");
  }
}

// geometry_msgs::Twist DynamicSystemTiago::calc_desired_base_transform(std::vector<double> &base_actions,
//                                                                     tf::Vector3 planned_base_vel,
//                                                                     tf::Quaternion planned_base_q,
//                                                                     tf::Vector3 planned_gripper_vel,
//                                                                     tf::Transform &desiredBaseTransform,
//                                                                     double transition_noise_base,
//                                                                     double &regularization,
//                                                                     const double &last_dt,
//                                                                     const tf::Transform &desiredGripperTransform) {
//    // a) calculate the new desire baseTransform
//    // planner actions are based on last_dt, RL actions are for a unit time -> scale down RL actions
//    double base_rot_rng_t = last_dt * robo_config_.base_rot_rng;
//    double base_vel_rng_t = last_dt * robo_config_.base_vel_rng;
//    double vel_forward, angle;
//
//    if (strategy_ == "dirvel") {
//        vel_forward = base_vel_rng_t * base_actions[0];
//        angle = base_rot_rng_t * base_actions[1];
//        regularization += pow(base_actions[0], 2.0) + pow(base_actions[1], 2.0);
//    } else {
//        // velocity we want to achieve, then apply the motion constraints to it
//        tf::Vector3 goal_vel_vec;
//        double dvel;
//        double dangle;
//        if ((strategy_ == "relvelm") || (strategy_ == "relveld")) {
//            dvel = base_vel_rng_t * base_actions[0];
//            dangle = base_rot_rng_t * base_actions[1];
//            goal_vel_vec = tf::Vector3(planned_gripper_vel.x(), planned_gripper_vel.y(), 0.0);
//        } else if (strategy_ == "unmodulated") {
//            dvel = 0.0;
//            dangle = 0.0;
//            // goal_vel_vec = tf::Vector3(planned_gripper_vel.x(), planned_gripper_vel.y(), 0.0);
//            goal_vel_vec = tf::Vector3(planned_base_vel.x(), planned_base_vel.y(), 0.0);
//        } else {
//            throw std::runtime_error("Unimplemented strategy for tiago");
//        }
//
//        vel_forward = goal_vel_vec.length();
//        if (goal_vel_vec.x() < 0.0) {
//            vel_forward = -vel_forward;
//        }
//        vel_forward += dvel;
//
//        // angle to rotate by as angle between just moving forward and the gripper movement
//        tf::Vector3 vel_forward_vec = tf::Vector3(vel_forward, 0.0, 0.0);
//        angle = atan2(vel_forward_vec.x() * goal_vel_vec.y() - vel_forward_vec.y() * goal_vel_vec.x(),
//        vel_forward_vec.dot(goal_vel_vec)); angle = utils::clamp_double(angle, -base_rot_rng_t, base_rot_rng_t); angle +=
//        dangle;
//
//        // unmodulated alternative: first rotate then move
//        // if (strategy_ == "unmodulated"){
//        //     if (std::abs(angle) > 0.01){
//        //         vel_forward = 0.0;
//        //         angle = utils::clamp_double(angle, -0.1, 0.1);
//        //      } else {
//        //         angle = 0.0;
//        //      }
//        // }
//
//        // enforce constraints
//        vel_forward = utils::clamp_double(vel_forward, -base_vel_rng_t, base_vel_rng_t);
//        angle = utils::clamp_double(angle, -base_rot_rng_t, base_rot_rng_t);
//
//        if (strategy_ == "relvelm") {
//            // a) modulate as little as possible
//            regularization += pow(base_actions[0], 2.0) + pow(base_actions[1], 2.0);
//        } else {
//            // b) keep total speed low (scaled back up into -1, 1 range)
//            double denom = std::abs(base_vel_rng_t) < 0.000001 ? 1.0 : base_vel_rng_t;
//            regularization += pow(base_actions[0], 1) + pow(vel_forward / base_vel_rng_t, 2);
//        }
//    }
//
//    if (transition_noise_base > 0.0001) {
//        vel_forward += rng_.gaussian(0.0, transition_noise_base);
//        angle += rng_.gaussian(0.0, transition_noise_base);
//    }
//
//    // calculate base transform after applying the new velocities
//    desiredBaseTransform = currentBaseTransform_;
//    tf::Quaternion q(tf::Vector3(0.0, 0.0, 1.0), angle);
//    desiredBaseTransform.setRotation(q * currentBaseTransform_.getRotation());
//
//    // world reference frame: move relative to base_angle defined by now rotated base
//    double base_angle = desiredBaseTransform.getRotation().getAngle() * desiredBaseTransform.getRotation().getAxis().getZ();
//    tf::Vector3 base_velocity_dif_drive(vel_forward * cos(base_angle), vel_forward * sin(base_angle), 0.0);
//    desiredBaseTransform.setOrigin(currentBaseTransform_.getOrigin() + base_velocity_dif_drive);
//
//    // calculate corresponding cmd, scaled up to a unit time again
//    double cmd_scaling = 1.0;
//    if (last_dt > 0.001) {
//        cmd_scaling /= last_dt;
//    }
//    geometry_msgs::Twist base_cmd_rel;
//    base_cmd_rel.linear.x = vel_forward * cmd_scaling;
//    base_cmd_rel.angular.z = angle * cmd_scaling;
//
//    return base_cmd_rel;
//}

// Moveit does not seem capable of updating the plan each iteration, rather just cancels it or similar
// void DynamicSystemTiago::send_arm_command(const std::vector<double> &target_joint_values, double exec_duration){
//     move_group_arm_torso_->setStartStateToCurrentState();
//
//    for(int i=0; i<joint_names_.size();i++){
//        // ROS_INFO_STREAM("\t" << joint_names_[i] << " goal position: " << target_joint_values[i]);
//        move_group_arm_torso_->setJointValueTarget(joint_names_[i], target_joint_values[i]);
//    }
//
//    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
//    //move_group_arm_torso_->setPlanningTime(5.0);
//    bool success = true;
//    success = (move_group_arm_torso_->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
//    if ( !success )
//        ROS_WARN("No arm_torso plan found");
//    else {
//        // ROS_INFO_STREAM("Plan found in " << my_plan.planning_time_ << " seconds");
//        move_group_arm_torso_->asyncExecute(my_plan);
//    }
////    return success;
//}

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

// void DynamicSystemTiago::stop_controllers(){
//    controllers will try to return to previous pose -> stop and restart
//    pr2_mechanism_msgs::SwitchController stop;
//    stop.request.stop_controllers.push_back("arm_controller");
//    stop.request.stop_controllers.push_back("torso_controller");
//    stop.request.stop_controllers.push_back("gripper_controller");
//    if (!switch_controller_client_.call(stop)) {
//        ROS_INFO("switch_controller_client_ failed at stop");
//    };
//}
//
// void DynamicSystemTiago::start_controllers(){
//    pr2_mechanism_msgs::SwitchController start;
//    start.request.start_controllers.push_back("arm_controller");
//    start.request.start_controllers.push_back("torso_controller");
//    start.request.start_controllers.push_back("gripper_controller");
//    if (!switch_controller_client_.call(start)) {
//        ROS_INFO("switch_controller_client_ failed at start");
//    };
//}
