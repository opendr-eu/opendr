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

#include <mobile_manipulation_rl/worlds.hpp>

BaseWorld::BaseWorld(std::string name, bool is_analytical) : name_{name}, is_analytical_{is_analytical} {
  if (name_ != "sim") {
    listener_.waitForTransform("map", "base_footprint", ros::Time(0), ros::Duration(10.0));
  }
};

tf::Transform BaseWorld::getBaseTransformWorld() {
  if (name_ != "sim") {
    tf::StampedTransform newBaseTransform;
    listener_.lookupTransform("map", "base_footprint", ros::Time(0), newBaseTransform);
    // Seems to sometimes return a non-zero z coordinate for e.g. PR2
    newBaseTransform.setOrigin(tf::Vector3(newBaseTransform.getOrigin().x(), newBaseTransform.getOrigin().y(), 0.0));
    return tf::Transform(newBaseTransform);
  } else {
    throw std::runtime_error("Not implemented for this world type: " + name_);
  }
}

SimWorld::SimWorld() : BaseWorld("sim", true){};

void SimWorld::setModelState(std::string model_name, tf::Transform world_transform, RoboConf robo_config,
                             ros::Publisher &cmd_base_vel_pub) {
}

GazeboWorld::GazeboWorld() : BaseWorld("gazebo", false){};

void GazeboWorld::setModelState(std::string model_name, tf::Transform world_transform, RoboConf robo_config,
                                ros::Publisher &cmd_base_vel_pub) {
  // NOTE: controllers might try to return to previous pose -> stop and restart within inherited class if necessary
  // stop_controllers();

  // pause physics
  std_srvs::Empty emptySrv;
  ros::service::call("/gazebo/pause_physics", emptySrv);
  // set base in gazebo
  gazebo_msgs::ModelState modelstate;
  modelstate.model_name = model_name;
  modelstate.reference_frame = "world";
  modelstate.pose.position.x = world_transform.getOrigin().x();
  modelstate.pose.position.y = world_transform.getOrigin().y();
  modelstate.pose.position.z = world_transform.getOrigin().z();
  modelstate.pose.orientation.x = world_transform.getRotation().x();
  modelstate.pose.orientation.y = world_transform.getRotation().y();
  modelstate.pose.orientation.z = world_transform.getRotation().z();
  modelstate.pose.orientation.w = world_transform.getRotation().w();

  gazebo_msgs::SetModelState srv;
  srv.request.model_state = modelstate;

  if (!ros::service::call("/gazebo/set_model_state", srv)) {
    ROS_ERROR("set_model_state_client_ failed");
  };

  // unpause physics
  ros::service::call("/gazebo/unpause_physics", emptySrv);

  // start_controllers();
}

RealWorld::RealWorld() : BaseWorld("world", false){};

bool RealWorld::isWithinWorld(tf::Transform base_transform) {
  double min_x = -0.0, max_x = 3.5, min_y = -0.0, max_y = 2.0, max_y_small = 1.0;

  tf::Vector3 g = base_transform.getOrigin();
  bool valid = (g.x() >= min_x) && (g.x() <= max_x) && (g.y() >= min_y) && (g.y() <= max_y);
  if (g.x() <= 1.0) {
    valid &= (g.y() <= max_y_small);
  }
  return valid;
}

void RealWorld::setModelState(std::string model_name, tf::Transform world_transform, RoboConf robo_config,
                              ros::Publisher &cmd_base_vel_pub) {
  if (model_name != "pr2") {
    throw std::runtime_error("ONLY IMPLEMENTED FOR PR2 SO FAR");
  }
  ros::Rate rate(50);

  tf::Transform currentBaseTransform = getBaseTransformWorld();
  tf::Vector3 goal_vec = world_transform.getOrigin() - currentBaseTransform.getOrigin();

  while (goal_vec.length() > 0.025) {
    // location
    tf::Vector3 scaled_vel = utils::normScaleVel(goal_vec, 0.0, robo_config.base_vel_rng);
    tf::Transform desiredBaseTransform = currentBaseTransform;
    desiredBaseTransform.setOrigin(currentBaseTransform.getOrigin() + scaled_vel);

    // rotation
    double roll_, pitch_, yaw_, yaw2_;
    tf::Matrix3x3(world_transform.getRotation()).getRPY(roll_, pitch_, yaw_);
    tf::Matrix3x3(currentBaseTransform.getRotation()).getRPY(roll_, pitch_, yaw2_);
    double angle_diff = utils::rpyAngleDiff(yaw_, yaw2_);
    double base_rotation = utils::clampDouble(angle_diff, -robo_config.base_rot_rng, robo_config.base_rot_rng);

    tf::Quaternion q(tf::Vector3(0.0, 0.0, 1.0), base_rotation);
    desiredBaseTransform.setRotation(q * currentBaseTransform.getRotation());

    // construct command
    tf::Transform relative_desired_pose = currentBaseTransform.inverse() * desiredBaseTransform;
    geometry_msgs::Twist base_cmd_rel;
    // double roll_, pitch_, yaw;
    relative_desired_pose.getBasis().getRPY(roll_, pitch_, yaw_);
    base_cmd_rel.linear.x = relative_desired_pose.getOrigin().getX();
    base_cmd_rel.linear.y = relative_desired_pose.getOrigin().getY();
    base_cmd_rel.angular.z = yaw_;

    // publish command
    cmd_base_vel_pub.publish(base_cmd_rel);
    rate.sleep();

    // update
    currentBaseTransform = getBaseTransformWorld();
    goal_vec = world_transform.getOrigin() - currentBaseTransform.getOrigin();
  }
}
