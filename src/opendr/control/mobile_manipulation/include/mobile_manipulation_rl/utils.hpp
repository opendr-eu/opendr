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

#ifndef UTILS_H
#define UTILS_H

#include <eigen_conversions/eigen_msg.h>
#include <math.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <sstream>
#include "std_msgs/ColorRGBA.h"
#include "tf/transform_datatypes.h"
#include "visualization_msgs/MarkerArray.h"
#include "yaml-cpp/yaml.h"

typedef std::map<std::string, double> PathPoint;

struct RobotObs {
  const std::vector<double> base_tf;     // in world frame
  const std::vector<double> gripper_tf;  // in world frame
  const std::vector<double> relative_gripper_tf;
  const std::vector<double> joint_values;
  const std::vector<double> base_velocity;
  double base_rotation_velocity;
  const std::vector<double> gripper_velocities;
  bool ik_fail;
  const double reward;
  const bool done;
};

namespace utils {
  tf::Vector3 qToRpy(tf::Quaternion q);
  double calcRotDist(const tf::Transform &a, const tf::Transform &b);
  double vec3AbsMax(tf::Vector3 v);
  visualization_msgs::Marker markerFromTransform(tf::Transform t, std::string ns, std_msgs::ColorRGBA color, int marker_id,
                                                 std::string frame_id, const std::string &geometry = "arrow",
                                                 tf::Vector3 marker_scale = tf::Vector3(0.1, 0.025, 0.025));
  std_msgs::ColorRGBA getColorMsg(const std::string &color_name, double alpha = 1.0);
  tf::Vector3 minMaxScaleVel(tf::Vector3 vel, double min_vel, double max_vel);
  tf::Vector3 normScaleVel(tf::Vector3 vel, double min_vel_norm, double max_vel_norm);
  tf::Vector3 maxClipVel(tf::Vector3 vel, double max_vel);
  double clampDouble(double value, double min_value, double max_value);
  tf::Transform tipToGripperGoal(const tf::Transform &gripperTipGoalWorld, const tf::Vector3 &tip_to_gripper_offset,
                                 const tf::Quaternion &gripper_to_base_rot_offset);
  tf::Transform gripperToTipGoal(const tf::Transform &gripperWristGoalWorld, const tf::Vector3 &tip_to_gripper_offset,
                                 const tf::Quaternion &gripper_to_base_rot_offset);
  double rpyAngleDiff(double next, double prev);
  bool startsWith(const std::string &str, const std::string &substr);
  bool endsWith(const std::string &str, const std::string &substr);
  std::string trim(const std::string &s);
  tf::Transform listToTf(const std::vector<double> &input);
  std::vector<double> tfToList(const tf::Transform &input, bool normalize_q = false);
  std::vector<double> vector3ToList(const tf::Vector3 &input);
  tf::Vector3 listToVector3(const std::vector<double> &input);
  std::vector<double> quaternionToList(const tf::Quaternion &input, bool normalize_q = false);
  tf::Quaternion listToQuaternion(const std::vector<double> &input);
  tf::Quaternion calcDq(tf::Quaternion current, tf::Quaternion next);
  bool tfAlmostEqual(tf::Transform a, tf::Transform b);
  std::vector<double> pythonMultiplyTfs(const std::vector<double> &tf1_list, const std::vector<double> &tf2_list,
                                        bool invert_tf1);
}  // namespace utils

struct RoboConf {
  const std::string name;
  const std::string joint_model_group_name;
  const std::string frame_id;
  const std::string global_link_transform;
  const std::string scene_collision_group_name;
  const tf::Vector3 tip_to_gripper_offset;
  const tf::Quaternion gripper_to_base_rot_offset;
  const std::vector<std::string> neutral_pos_joint_names;
  const std::vector<double> neutral_pos_values;
  const std::string base_cmd_topic;
  const double base_vel_rng;
  const double base_rot_rng;
  const double z_min;
  const double z_max;
  const double restricted_ws_z_min;
  const double restricted_ws_z_max;
  const double gmm_base_offset;
  const double kinematics_solver_timeout;
  const double robot_base_size_meters_x;
  const double robot_base_size_meters_y;
  std::map<std::string, double> initial_joint_values;
  std::tuple<std::map<std::string, std::vector<double>>, std::map<std::string, std::string>> toPython() {
    std::map<std::string, std::vector<double>> m;
    m["z_min"] = std::vector<double>{z_min};
    m["z_max"] = std::vector<double>{z_max};
    m["restricted_ws_z_min"] = std::vector<double>{restricted_ws_z_min};
    m["restricted_ws_z_max"] = std::vector<double>{restricted_ws_z_max};
    m["tip_to_gripper_offset"] = utils::vector3ToList(tip_to_gripper_offset);
    m["gripper_to_base_rot_offset"] = utils::quaternionToList(gripper_to_base_rot_offset);
    m["base_vel_rng"] = std::vector<double>{base_vel_rng};
    m["gmm_base_offset"] = std::vector<double>{gmm_base_offset};
    m["robot_base_size_meters_x"] = std::vector<double>{robot_base_size_meters_x};
    m["robot_base_size_meters_y"] = std::vector<double>{robot_base_size_meters_y};

    std::map<std::string, std::string> m_str;
    m_str["frame_id"] = frame_id;

    return std::tuple<std::map<std::string, std::vector<double>>, std::map<std::string, std::string>>{m, m_str};
  };

  static RoboConf fromYaml(const std::string &robo_conf_path) {
    std::ifstream f(robo_conf_path.c_str());
    if (!f.good()) {
      throw std::runtime_error("robot_config.yaml not found. Please run from project root.");
    }

    YAML::Node config = YAML::LoadFile(robo_conf_path);

    RoboConf robot_config{.name = config["name"].as<std::string>(),
                          .joint_model_group_name = config["joint_model_group_name"].as<std::string>(),
                          .frame_id = config["frame_id"].as<std::string>(),
                          .global_link_transform = config["global_link_transform"].as<std::string>(),
                          .scene_collision_group_name = config["scene_collision_group_name"].as<std::string>(),
                          .tip_to_gripper_offset = tf::Vector3(config["tip_to_gripper_offset"]["x"].as<double>(),
                                                               config["tip_to_gripper_offset"]["y"].as<double>(),
                                                               config["tip_to_gripper_offset"]["z"].as<double>()),
                          .gripper_to_base_rot_offset = tf::Quaternion(config["gripper_to_base_rot_offset"]["x"].as<double>(),
                                                                       config["gripper_to_base_rot_offset"]["y"].as<double>(),
                                                                       config["gripper_to_base_rot_offset"]["z"].as<double>(),
                                                                       config["gripper_to_base_rot_offset"]["w"].as<double>()),
                          .neutral_pos_joint_names = config["neutral_pos_joint_names"].as<std::vector<std::string>>(),
                          .neutral_pos_values = config["neutral_pos_values"].as<std::vector<double>>(),
                          .base_cmd_topic = config["base_cmd_topic"].as<std::string>(),
                          .base_vel_rng = config["base_vel_rng"].as<double>(),
                          .base_rot_rng = config["base_rot_rng"].as<double>(),
                          .z_min = config["z_min"].as<double>(),
                          .z_max = config["z_max"].as<double>(),
                          .restricted_ws_z_min = config["restricted_ws_z_min"].as<double>(),
                          .restricted_ws_z_max = config["restricted_ws_z_max"].as<double>(),
                          .gmm_base_offset = config["gmm_base_offset"].as<double>(),
                          .kinematics_solver_timeout = config["kinematics_solver_timeout"].as<double>(),
                          .robot_base_size_meters_x = config["robot_base_size_meters"]["x"].as<double>(),
                          .robot_base_size_meters_y = config["robot_base_size_meters"]["y"].as<double>(),
                          .initial_joint_values = (config["initial_joint_values"]) ?
                                                    config["initial_joint_values"].as<std::map<std::string, double>>() :
                                                    std::map<std::string, double>()};
    return robot_config;
  }
};

#endif