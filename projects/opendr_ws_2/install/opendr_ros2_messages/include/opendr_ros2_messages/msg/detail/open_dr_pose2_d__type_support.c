// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from opendr_ros2_messages:msg/OpenDRPose2D.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d__rosidl_typesupport_introspection_c.h"
#include "opendr_ros2_messages/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d__functions.h"
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `keypoint_list`
#include "opendr_ros2_messages/msg/open_dr_pose2_d_keypoint.h"
// Member `keypoint_list`
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  opendr_ros2_messages__msg__OpenDRPose2D__init(message_memory);
}

void OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_fini_function(void * message_memory)
{
  opendr_ros2_messages__msg__OpenDRPose2D__fini(message_memory);
}

size_t OpenDRPose2D__rosidl_typesupport_introspection_c__size_function__OpenDRPose2DKeypoint__keypoint_list(
  const void * untyped_member)
{
  const opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * member =
    (const opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence *)(untyped_member);
  return member->size;
}

const void * OpenDRPose2D__rosidl_typesupport_introspection_c__get_const_function__OpenDRPose2DKeypoint__keypoint_list(
  const void * untyped_member, size_t index)
{
  const opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * member =
    (const opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence *)(untyped_member);
  return &member->data[index];
}

void * OpenDRPose2D__rosidl_typesupport_introspection_c__get_function__OpenDRPose2DKeypoint__keypoint_list(
  void * untyped_member, size_t index)
{
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * member =
    (opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence *)(untyped_member);
  return &member->data[index];
}

bool OpenDRPose2D__rosidl_typesupport_introspection_c__resize_function__OpenDRPose2DKeypoint__keypoint_list(
  void * untyped_member, size_t size)
{
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * member =
    (opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence *)(untyped_member);
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__fini(member);
  return opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_member_array[4] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__msg__OpenDRPose2D, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "pose_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__msg__OpenDRPose2D, pose_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "conf",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__msg__OpenDRPose2D, conf),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "keypoint_list",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__msg__OpenDRPose2D, keypoint_list),  // bytes offset in struct
    NULL,  // default value
    OpenDRPose2D__rosidl_typesupport_introspection_c__size_function__OpenDRPose2DKeypoint__keypoint_list,  // size() function pointer
    OpenDRPose2D__rosidl_typesupport_introspection_c__get_const_function__OpenDRPose2DKeypoint__keypoint_list,  // get_const(index) function pointer
    OpenDRPose2D__rosidl_typesupport_introspection_c__get_function__OpenDRPose2DKeypoint__keypoint_list,  // get(index) function pointer
    OpenDRPose2D__rosidl_typesupport_introspection_c__resize_function__OpenDRPose2DKeypoint__keypoint_list  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_members = {
  "opendr_ros2_messages__msg",  // message namespace
  "OpenDRPose2D",  // message name
  4,  // number of fields
  sizeof(opendr_ros2_messages__msg__OpenDRPose2D),
  OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_member_array,  // message members
  OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_init_function,  // function to initialize message memory (memory has to be allocated)
  OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_type_support_handle = {
  0,
  &OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_opendr_ros2_messages
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, msg, OpenDRPose2D)() {
  OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, msg, OpenDRPose2DKeypoint)();
  if (!OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_type_support_handle.typesupport_identifier) {
    OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &OpenDRPose2D__rosidl_typesupport_introspection_c__OpenDRPose2D_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
