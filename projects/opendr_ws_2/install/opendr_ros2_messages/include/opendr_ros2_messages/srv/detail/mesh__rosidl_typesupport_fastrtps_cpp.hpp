// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from opendr_ros2_messages:srv/Mesh.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "opendr_ros2_messages/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "opendr_ros2_messages/srv/detail/mesh__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace opendr_ros2_messages
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
cdr_serialize(
  const opendr_ros2_messages::srv::Mesh_Request & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  opendr_ros2_messages::srv::Mesh_Request & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
get_serialized_size(
  const opendr_ros2_messages::srv::Mesh_Request & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
max_serialized_size_Mesh_Request(
  bool & full_bounded,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace opendr_ros2_messages

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, opendr_ros2_messages, srv, Mesh_Request)();

#ifdef __cplusplus
}
#endif

// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "opendr_ros2_messages/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
// already included above
// #include "opendr_ros2_messages/srv/detail/mesh__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// already included above
// #include "fastcdr/Cdr.h"

namespace opendr_ros2_messages
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
cdr_serialize(
  const opendr_ros2_messages::srv::Mesh_Response & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  opendr_ros2_messages::srv::Mesh_Response & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
get_serialized_size(
  const opendr_ros2_messages::srv::Mesh_Response & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
max_serialized_size_Mesh_Response(
  bool & full_bounded,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace opendr_ros2_messages

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, opendr_ros2_messages, srv, Mesh_Response)();

#ifdef __cplusplus
}
#endif

#include "rmw/types.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "opendr_ros2_messages/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_opendr_ros2_messages
const rosidl_service_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, opendr_ros2_messages, srv, Mesh)();

#ifdef __cplusplus
}
#endif

#endif  // OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
