// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from opendr_ros2_messages:srv/Mesh.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__STRUCT_H_
#define OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'rgb_img'
// Member 'msk_img'
#include "sensor_msgs/msg/detail/image__struct.h"
// Member 'extract_pose'
#include "std_msgs/msg/detail/bool__struct.h"

// Struct defined in srv/Mesh in the package opendr_ros2_messages.
typedef struct opendr_ros2_messages__srv__Mesh_Request
{
  sensor_msgs__msg__Image rgb_img;
  sensor_msgs__msg__Image msk_img;
  std_msgs__msg__Bool extract_pose;
} opendr_ros2_messages__srv__Mesh_Request;

// Struct for a sequence of opendr_ros2_messages__srv__Mesh_Request.
typedef struct opendr_ros2_messages__srv__Mesh_Request__Sequence
{
  opendr_ros2_messages__srv__Mesh_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} opendr_ros2_messages__srv__Mesh_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'mesh'
#include "shape_msgs/msg/detail/mesh__struct.h"
// Member 'vertex_colors'
#include "std_msgs/msg/detail/color_rgba__struct.h"
// Member 'pose'
#include "vision_msgs/msg/detail/detection3_d_array__struct.h"

// Struct defined in srv/Mesh in the package opendr_ros2_messages.
typedef struct opendr_ros2_messages__srv__Mesh_Response
{
  shape_msgs__msg__Mesh mesh;
  std_msgs__msg__ColorRGBA__Sequence vertex_colors;
  vision_msgs__msg__Detection3DArray pose;
} opendr_ros2_messages__srv__Mesh_Response;

// Struct for a sequence of opendr_ros2_messages__srv__Mesh_Response.
typedef struct opendr_ros2_messages__srv__Mesh_Response__Sequence
{
  opendr_ros2_messages__srv__Mesh_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} opendr_ros2_messages__srv__Mesh_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__STRUCT_H_
