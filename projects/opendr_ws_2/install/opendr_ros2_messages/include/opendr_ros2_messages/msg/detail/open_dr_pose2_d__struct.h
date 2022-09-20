// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from opendr_ros2_messages:msg/OpenDRPose2D.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__STRUCT_H_
#define OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'keypoint_list'
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__struct.h"

// Struct defined in msg/OpenDRPose2D in the package opendr_ros2_messages.
typedef struct opendr_ros2_messages__msg__OpenDRPose2D
{
  std_msgs__msg__Header header;
  int32_t pose_id;
  float conf;
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence keypoint_list;
} opendr_ros2_messages__msg__OpenDRPose2D;

// Struct for a sequence of opendr_ros2_messages__msg__OpenDRPose2D.
typedef struct opendr_ros2_messages__msg__OpenDRPose2D__Sequence
{
  opendr_ros2_messages__msg__OpenDRPose2D * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} opendr_ros2_messages__msg__OpenDRPose2D__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__STRUCT_H_
