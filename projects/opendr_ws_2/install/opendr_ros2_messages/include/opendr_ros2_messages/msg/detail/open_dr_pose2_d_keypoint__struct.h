// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from opendr_ros2_messages:msg/OpenDRPose2DKeypoint.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__STRUCT_H_
#define OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'kpt_name'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/OpenDRPose2DKeypoint in the package opendr_ros2_messages.
typedef struct opendr_ros2_messages__msg__OpenDRPose2DKeypoint
{
  rosidl_runtime_c__String kpt_name;
  int32_t x;
  int32_t y;
} opendr_ros2_messages__msg__OpenDRPose2DKeypoint;

// Struct for a sequence of opendr_ros2_messages__msg__OpenDRPose2DKeypoint.
typedef struct opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence
{
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__STRUCT_H_
