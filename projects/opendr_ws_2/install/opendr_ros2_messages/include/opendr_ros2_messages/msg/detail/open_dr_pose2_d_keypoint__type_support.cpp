// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from opendr_ros2_messages:msg/OpenDRPose2DKeypoint.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace opendr_ros2_messages
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void OpenDRPose2DKeypoint_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) opendr_ros2_messages::msg::OpenDRPose2DKeypoint(_init);
}

void OpenDRPose2DKeypoint_fini_function(void * message_memory)
{
  auto typed_message = static_cast<opendr_ros2_messages::msg::OpenDRPose2DKeypoint *>(message_memory);
  typed_message->~OpenDRPose2DKeypoint();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember OpenDRPose2DKeypoint_message_member_array[3] = {
  {
    "kpt_name",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::msg::OpenDRPose2DKeypoint, kpt_name),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "x",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::msg::OpenDRPose2DKeypoint, x),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "y",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::msg::OpenDRPose2DKeypoint, y),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers OpenDRPose2DKeypoint_message_members = {
  "opendr_ros2_messages::msg",  // message namespace
  "OpenDRPose2DKeypoint",  // message name
  3,  // number of fields
  sizeof(opendr_ros2_messages::msg::OpenDRPose2DKeypoint),
  OpenDRPose2DKeypoint_message_member_array,  // message members
  OpenDRPose2DKeypoint_init_function,  // function to initialize message memory (memory has to be allocated)
  OpenDRPose2DKeypoint_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t OpenDRPose2DKeypoint_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &OpenDRPose2DKeypoint_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace opendr_ros2_messages


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<opendr_ros2_messages::msg::OpenDRPose2DKeypoint>()
{
  return &::opendr_ros2_messages::msg::rosidl_typesupport_introspection_cpp::OpenDRPose2DKeypoint_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, opendr_ros2_messages, msg, OpenDRPose2DKeypoint)() {
  return &::opendr_ros2_messages::msg::rosidl_typesupport_introspection_cpp::OpenDRPose2DKeypoint_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
