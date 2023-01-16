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

#include <mobile_manipulation_rl/utils.hpp>

namespace utils {
  tf::Vector3 qToRpy(tf::Quaternion q) {
    double roll, pitch, yaw;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return tf::Vector3(roll, pitch, yaw);
  }

  double calcRotDist(const tf::Transform &a, const tf::Transform &b) {
    double inner_prod = a.getRotation().dot(b.getRotation());
    return 1.0 - pow(inner_prod, 2.0);
  }

  double vec3AbsMax(tf::Vector3 v) {
    tf::Vector3 v_abs = v.absolute();
    return std::max(std::max(v_abs.x(), v_abs.y()), v_abs.z());
  }

  visualization_msgs::Marker markerFromTransform(tf::Transform t, std::string ns, std_msgs::ColorRGBA color, int marker_id,
                                                 std::string frame_id, const std::string &geometry, tf::Vector3 marker_scale) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = std::move(frame_id);
    marker.header.stamp = ros::Time();
    marker.ns = std::move(ns);

    if (geometry == "arrow") {
      marker.type = visualization_msgs::Marker::ARROW;
    } else if (geometry == "cube") {
      marker.type = visualization_msgs::Marker::CUBE;
    } else {
      throw std::runtime_error("unknown marker geometry");
    }
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = t.getOrigin().x();
    marker.pose.position.y = t.getOrigin().y();
    marker.pose.position.z = t.getOrigin().z();
    marker.pose.orientation.x = t.getRotation().x();
    marker.pose.orientation.y = t.getRotation().y();
    marker.pose.orientation.z = t.getRotation().z();
    marker.pose.orientation.w = t.getRotation().w();
    marker.scale.x = marker_scale.x();
    marker.scale.y = marker_scale.y();
    marker.scale.z = marker_scale.z();

    // more and more red from 0 to 100
    marker.color = color;
    marker.id = marker_id;
    return marker;
  }

  std_msgs::ColorRGBA getColorMsg(const std::string &color_name, double alpha) {
    std_msgs::ColorRGBA c;
    if (color_name == "blue") {
      c.b = 1.0;
    } else if (color_name == "pink") {
      c.r = 1.0;
      c.g = 105.0 / 255.0;
      c.b = 147.0 / 255.0;
    } else if (color_name == "orange") {
      c.r = 1.0;
      c.g = 159.0 / 255.0;
      c.b = 0.0;
    } else if (color_name == "yellow") {
      c.r = 1.0;
      c.g = 1.0;
      c.b = 0.0;
    } else if (color_name == "cyan") {
      c.r = 0.0;
      c.g = 128.0 / 255.0;
      c.b = 1.0;
    } else if (color_name == "green") {
      c.r = 0.0;
      c.g = 1.0;
      c.b = 0.0;
    } else if (color_name == "red") {
      c.r = 1.0;
      c.g = 0.0;
      c.b = 0.0;
    } else {
      throw std::runtime_error("unknown color");
    }
    c.a = alpha;
    return c;
  }

  tf::Vector3 minMaxScaleVel(tf::Vector3 vel, double min_vel, double max_vel) {
    // find denominator to keep it in range [min_planner_velocity_, max_planner_velocity_]
    double max_abs_vector_value = utils::vec3AbsMax(vel);
    // in case vel is a vector of all zeros avoid division by zero
    if (max_abs_vector_value == 0.0) {
      return tf::Vector3(min_vel, min_vel, min_vel);
    }
    double max_denom;
    if (min_vel < 0.001) {
      max_denom = 1.0;
    } else {
      max_denom = std::min(max_abs_vector_value / min_vel, 1.0);
    }
    double min_denom = max_abs_vector_value / max_vel;
    double denom = std::max(max_denom, min_denom);
    return vel / denom;
  }

  tf::Vector3 maxClipVel(tf::Vector3 vel, double max_vel) {
    tf::Vector3 clipped_vel;
    clipped_vel.setX(std::max(std::min(vel.x(), max_vel), -max_vel));
    clipped_vel.setY(std::max(std::min(vel.y(), max_vel), -max_vel));
    clipped_vel.setZ(std::max(std::min(vel.z(), max_vel), -max_vel));
    return clipped_vel;
  }

  tf::Vector3 normScaleVel(tf::Vector3 vel, double min_vel_norm, double max_vel_norm) {
    double norm = vel.length();
    if (norm == 0.0) {
      return vel;
    } else if (max_vel_norm == 0.0) {
      return tf::Vector3(0.0, 0.0, 0.0);
    } else {
      double max_denom;
      if (min_vel_norm < 0.00000001) {
        max_denom = 1.0;
      } else {
        max_denom = std::min(norm / min_vel_norm, 1.0);
      }
      double min_denom = norm / max_vel_norm;
      double denom = std::max(max_denom, min_denom);

      //        assert((vel / denom).length() >= min_vel_norm - 0.001);
      //        assert((vel / denom).length() <= max_vel_norm + 0.001);

      return vel / denom;
    }
  }

  double clampDouble(double value, double min_value, double max_value) {
    return std::max(std::min(value, max_value), min_value);
  }

  tf::Transform tipToGripperGoal(const tf::Transform &gripperTipGoalWorld, const tf::Vector3 &tip_to_gripper_offset,
                                 const tf::Quaternion &gripper_to_base_rot_offset) {
    // gripper tip offset from wrist
    tf::Transform goal_no_trans(gripperTipGoalWorld);
    goal_no_trans.setOrigin(tf::Vector3(0, 0, 0));
    tf::Vector3 offset_pos = goal_no_trans * tip_to_gripper_offset;

    tf::Transform gripper_goal_wrist_world(gripperTipGoalWorld);
    gripper_goal_wrist_world.setOrigin(gripper_goal_wrist_world.getOrigin() - offset_pos);

    // different rotations between gripper joint and base/world
    gripper_goal_wrist_world.setRotation((gripper_goal_wrist_world.getRotation() * gripper_to_base_rot_offset).normalized());
    // utils::print_vector3(offset_pos, "offset_pos");
    // utils::print_q(gripper_to_base_rot_offset, "gripper_to_base_rot_offset");
    // utils::print_t(gripper_goal_wrist_world, "gripper_goal_wrist_world");
    return gripper_goal_wrist_world;
  }

  tf::Transform gripperToTipGoal(const tf::Transform &gripperWristGoalWorld, const tf::Vector3 &tip_to_gripper_offset,
                                 const tf::Quaternion &gripper_to_base_rot_offset) {
    tf::Transform gripper_goal_tip_world;
    gripper_goal_tip_world.setIdentity();

    // different rotations between gripper joint and base/world
    gripper_goal_tip_world.setRotation(gripperWristGoalWorld.getRotation() * gripper_to_base_rot_offset.inverse());

    // gripper tip offset from wrist
    tf::Vector3 offset_pos = gripper_goal_tip_world * tip_to_gripper_offset;
    gripper_goal_tip_world.setOrigin(gripperWristGoalWorld.getOrigin() + offset_pos);

    return gripper_goal_tip_world;
  }

  double rpyAngleDiff(double next, double prev) {
    double diff = next - prev;
    if (diff > M_PI) {
      diff = -2 * M_PI + diff;
    } else if (diff < -M_PI) {
      diff = 2 * M_PI + diff;
    }
    return diff;
  }

  bool startsWith(const std::string &str, const std::string &substr) { return (str.find(substr) == 0); }

  bool endsWith(const std::string &str, const std::string &substr) {
    size_t pos = str.rfind(substr);
    if (pos == std::string::npos)  // doesnt even contain it
      return false;

    size_t len = str.length();
    size_t elen = substr.length();
    // at end means: Pos found + length of end equal length of full string.
    if (pos + elen == len) {
      return true;
    }

    // not at end
    return false;
  }

  std::string trim(const std::string &s) {
    if (s.length() == 0)
      return s;
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    if (b == std::string::npos)
      return "";
    return std::string(s, b, e - b + 1);
  }

  tf::Transform listToTf(const std::vector<double> &input) {
    tf::Quaternion rotation;
    if (input.size() == 6) {
      rotation.setRPY(input[3], input[4], input[5]);
    } else if (input.size() == 7) {
      rotation = tf::Quaternion(input[3], input[4], input[5], input[6]);
    } else {
      throw std::runtime_error("invalid length of specified gripper goal");
    }
    return tf::Transform(rotation, tf::Vector3(input[0], input[1], input[2]));
  }

  std::vector<double> tfToList(const tf::Transform &input, bool normalize_q) {
    std::vector<double> output;
    output.push_back(input.getOrigin().x());
    output.push_back(input.getOrigin().y());
    output.push_back(input.getOrigin().z());
    tf::Quaternion q(input.getRotation());
    if (normalize_q) {
      q.normalize();
    }
    output.push_back(q.x());
    output.push_back(q.y());
    output.push_back(q.z());
    output.push_back(q.w());
    return output;
  }

  std::vector<double> vector3ToList(const tf::Vector3 &input) {
    std::vector<double> output;
    output.push_back(input.x());
    output.push_back(input.y());
    output.push_back(input.z());
    return output;
  }

  tf::Vector3 listToVector3(const std::vector<double> &input) {
    if (input.size() == 3) {
      return tf::Vector3(input[0], input[1], input[2]);
    } else {
      throw std::runtime_error("invalid length of specified gripper goal");
    }
  }

  std::vector<double> quaternionToList(const tf::Quaternion &input, bool normalize_q) {
    std::vector<double> output;
    tf::Quaternion q(input);
    if (normalize_q) {
      q.normalize();
    }
    output.push_back(q.x());
    output.push_back(q.y());
    output.push_back(q.z());
    output.push_back(q.w());
    return output;
  }

  tf::Quaternion listToQuaternion(const std::vector<double> &input) {
    if (input.size() == 4) {
      return tf::Quaternion(input[0], input[1], input[2], input[3]);
    } else {
      throw std::runtime_error("invalid length of specified gripper goal");
    }
  }

  tf::Quaternion calcDq(tf::Quaternion current, tf::Quaternion next) {
    // planned change in rotation defined as dq * current == next
    return (next * current.inverse()).normalized();
  }

  bool tfAlmostEqual(tf::Transform a, tf::Transform b) {
    bool equal = (a.getOrigin() - b.getOrigin()).length() < 0.05;
    // NOTE: not sure if this is invariant to all equivalent quaternions
    equal &= (a.getRotation() - b.getRotation()).length() < 0.05;
    return equal;
  }

  std::vector<double> pythonMultiplyTfs(const std::vector<double> &tf1_list, const std::vector<double> &tf2_list,
                                        bool invert_tf1) {
    tf::Transform tf1 = listToTf(tf1_list);
    tf::Transform tf2 = listToTf(tf2_list);
    if (invert_tf1) {
      tf1 = tf1.inverse();
    }
    return tfToList(tf1 * tf2);
  }

}  // namespace utils
