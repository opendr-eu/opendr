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
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandTOL.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <ros/ros.h>
#include "std_msgs/Float64.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr &msg) {
  current_state = *msg;
}
float current_pose_x;
float current_pose_y;
float current_pose_z;

void current_pose_callback(const geometry_msgs::PoseStamped::ConstPtr &msg) {
  current_pose_x = msg->pose.position.x;
  current_pose_y = msg->pose.position.y;
  current_pose_z = msg->pose.position.z;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "way_point");
  ros::NodeHandle nh;

  ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);

  ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 100);
  ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
  ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
  // takeoff
  ros::ServiceClient takeoff_client = nh.serviceClient<mavros_msgs::CommandTOL>("mavros/cmd/takeoff");
  // drone pos sub
  ros::Subscriber drone_pos_sub =
    nh.subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 10, current_pose_callback);

  // the setpoint publishing rate MUST be faster than 2Hz
  ros::Rate rate(40.0);

  // wait for FCU connection
  while (ros::ok() && !current_state.connected) {
    ros::spinOnce();
    rate.sleep();
  }
  geometry_msgs::PoseStamped pose;
  pose.pose.position.x = 0;
  pose.pose.position.y = 0;
  pose.pose.position.z = 5;
  local_pos_pub.publish(pose);

  // send a few setpoints before starting
  for (int i = 100; ros::ok() && i > 0; --i) {
    local_pos_pub.publish(pose);
    ros::spinOnce();
    rate.sleep();
  }

  mavros_msgs::SetMode guided_set_mode;

  guided_set_mode.request.custom_mode = "GUIDED";

  mavros_msgs::CommandBool arm_cmd;
  arm_cmd.request.value = true;

  mavros_msgs::CommandTOL takeoff_cmd;

  takeoff_cmd.request.min_pitch = 0;
  takeoff_cmd.request.yaw = 0;
  takeoff_cmd.request.latitude = 0;
  takeoff_cmd.request.longitude = 0;
  takeoff_cmd.request.altitude = 5;

  set_mode_client.call(guided_set_mode);
  arming_client.call(arm_cmd);
  takeoff_client.call(takeoff_cmd);

  return 0;
}
