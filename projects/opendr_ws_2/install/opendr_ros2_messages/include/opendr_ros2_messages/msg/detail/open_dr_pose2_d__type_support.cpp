// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from opendr_ros2_messages:msg/OpenDRPose2D.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d__struct.hpp"
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

void OpenDRPose2D_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) opendr_ros2_messages::msg::OpenDRPose2D(_init);
}

void OpenDRPose2D_fini_function(void * message_memory)
{
  auto typed_message = static_cast<opendr_ros2_messages::msg::OpenDRPose2D *>(message_memory);
  typed_message->~OpenDRPose2D();
}

size_t size_function__OpenDRPose2D__keypoint_list(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<opendr_ros2_messages::msg::OpenDRPose2DKeypoint> *>(untyped_member);
  return member->size();
}

const void * get_const_function__OpenDRPose2D__keypoint_list(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<opendr_ros2_messages::msg::OpenDRPose2DKeypoint> *>(untyped_member);
  return &member[index];
}

void * get_function__OpenDRPose2D__keypoint_list(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<opendr_ros2_messages::msg::OpenDRPose2DKeypoint> *>(untyped_member);
  return &member[index];
}

void resize_function__OpenDRPose2D__keypoint_list(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<opendr_ros2_messages::msg::OpenDRPose2DKeypoint> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember OpenDRPose2D_message_member_array[4] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::msg::OpenDRPose2D, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "pose_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::msg::OpenDRPose2D, pose_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "conf",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::msg::OpenDRPose2D, conf),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "keypoint_list",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<opendr_ros2_messages::msg::OpenDRPose2DKeypoint>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::msg::OpenDRPose2D, keypoint_list),  // bytes offset in struct
    nullptr,  // default value
    size_function__OpenDRPose2D__keypoint_list,  // size() function pointer
    get_const_function__OpenDRPose2D__keypoint_list,  // get_const(index) function pointer
    get_function__OpenDRPose2D__keypoint_list,  // get(index) function pointer
    resize_function__OpenDRPose2D__keypoint_list  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers OpenDRPose2D_message_members = {
  "opendr_ros2_messages::msg",  // message namespace
  "OpenDRPose2D",  // message name
  4,  // number of fields
  sizeof(opendr_ros2_messages::msg::OpenDRPose2D),
  OpenDRPose2D_message_member_array,  // message members
  OpenDRPose2D_init_function,  // function to initialize message memory (memory has to be allocated)
  OpenDRPose2D_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t OpenDRPose2D_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &OpenDRPose2D_message_members,
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
get_message_type_support_handle<opendr_ros2_messages::msg::OpenDRPose2D>()
{
  return &::opendr_ros2_messages::msg::rosidl_typesupport_introspection_cpp::OpenDRPose2D_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, opendr_ros2_messages, msg, OpenDRPose2D)() {
  return &::opendr_ros2_messages::msg::rosidl_typesupport_introspection_cpp::OpenDRPose2D_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
