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

#ifndef MOBILE_MANIPULATION_RL_WORLDS_H
#define MOBILE_MANIPULATION_RL_WORLDS_H

#include <eigen_conversions/eigen_msg.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/SetModelConfiguration.h>
#include <gazebo_msgs/SetModelState.h>
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "tf/transform_datatypes.h"

#include <mobile_manipulation_rl/utils.hpp>

class BaseWorld {
public:
  BaseWorld(std::string name,
            // for worlds that do not include full simulation of controllers, continuous time, etc.
            bool is_analytical);
  const std::string name_;
  const bool is_analytical_;
  tf::TransformListener listener_;
  tf::Transform getBaseTransformWorld();
  virtual void setModelState(std::string model_name, tf::Transform world_transform, RoboConf robo_config,
                             ros::Publisher &cmd_base_vel_pub) = 0;

  std::string getName() const { return name_; };
  bool isAnalytical() const { return is_analytical_; };
  virtual bool isWithinWorld(tf::Transform base_transform) { return true; };
};

class GazeboWorld : public BaseWorld {
public:
  GazeboWorld();
  void setModelState(std::string model_name, tf::Transform world_transform, RoboConf robo_config,
                     ros::Publisher &cmd_base_vel_pub) override;
};

class RealWorld : public BaseWorld {
public:
  RealWorld();
  void setModelState(std::string model_name, tf::Transform world_transform, RoboConf robo_config,
                     ros::Publisher &cmd_base_vel_pub) override;
  bool isWithinWorld(tf::Transform base_transform) override;
};

class SimWorld : public BaseWorld {
public:
  SimWorld();
  void setModelState(std::string model_name, tf::Transform world_transform, RoboConf robo_config,
                     ros::Publisher &cmd_base_vel_pub) override;
};

#endif  // MOBILE_MANIPULATION_RL_WORLDS_H