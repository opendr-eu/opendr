// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from opendr_ros2_messages:msg/OpenDRPose2DKeypoint.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__rosidl_typesupport_introspection_c.h"
#include "opendr_ros2_messages/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__functions.h"
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__struct.h"


// Include directives for member types
// Member `kpt_name`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint__init(message_memory);
}

void OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_fini_function(void * message_memory)
{
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_message_member_array[3] = {
  {
    "kpt_name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__msg__OpenDRPose2DKeypoint, kpt_name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__msg__OpenDRPose2DKeypoint, x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__msg__OpenDRPose2DKeypoint, y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_message_members = {
  "opendr_ros2_messages__msg",  // message namespace
  "OpenDRPose2DKeypoint",  // message name
  3,  // number of fields
  sizeof(opendr_ros2_messages__msg__OpenDRPose2DKeypoint),
  OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_message_member_array,  // message members
  OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_init_function,  // function to initialize message memory (memory has to be allocated)
  OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_message_type_support_handle = {
  0,
  &OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_opendr_ros2_messages
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, msg, OpenDRPose2DKeypoint)() {
  if (!OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_message_type_support_handle.typesupport_identifier) {
    OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &OpenDRPose2DKeypoint__rosidl_typesupport_introspection_c__OpenDRPose2DKeypoint_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
