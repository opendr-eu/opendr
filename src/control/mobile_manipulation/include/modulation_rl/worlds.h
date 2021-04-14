#pragma once

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

#include <modulation_rl/utils.h>

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
private:
  //    ros::ServiceClient set_model_state_client_;
  //    ros::ServiceClient set_model_configuration_client_;
  //    ros::ServiceClient pause_gazebo_client_;
  //    ros::ServiceClient unpause_gazebo_client_;
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