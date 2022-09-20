// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from opendr_ros2_messages:srv/Mesh.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "opendr_ros2_messages/srv/detail/mesh__rosidl_typesupport_introspection_c.h"
#include "opendr_ros2_messages/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "opendr_ros2_messages/srv/detail/mesh__functions.h"
#include "opendr_ros2_messages/srv/detail/mesh__struct.h"


// Include directives for member types
// Member `rgb_img`
// Member `msk_img`
#include "sensor_msgs/msg/image.h"
// Member `rgb_img`
// Member `msk_img`
#include "sensor_msgs/msg/detail/image__rosidl_typesupport_introspection_c.h"
// Member `extract_pose`
#include "std_msgs/msg/bool.h"
// Member `extract_pose`
#include "std_msgs/msg/detail/bool__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  opendr_ros2_messages__srv__Mesh_Request__init(message_memory);
}

void Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_fini_function(void * message_memory)
{
  opendr_ros2_messages__srv__Mesh_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_member_array[3] = {
  {
    "rgb_img",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__srv__Mesh_Request, rgb_img),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "msk_img",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__srv__Mesh_Request, msk_img),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "extract_pose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__srv__Mesh_Request, extract_pose),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_members = {
  "opendr_ros2_messages__srv",  // message namespace
  "Mesh_Request",  // message name
  3,  // number of fields
  sizeof(opendr_ros2_messages__srv__Mesh_Request),
  Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_member_array,  // message members
  Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_type_support_handle = {
  0,
  &Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_opendr_ros2_messages
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, srv, Mesh_Request)() {
  Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, Image)();
  Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, Image)();
  Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Bool)();
  if (!Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_type_support_handle.typesupport_identifier) {
    Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &Mesh_Request__rosidl_typesupport_introspection_c__Mesh_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "opendr_ros2_messages/srv/detail/mesh__rosidl_typesupport_introspection_c.h"
// already included above
// #include "opendr_ros2_messages/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "opendr_ros2_messages/srv/detail/mesh__functions.h"
// already included above
// #include "opendr_ros2_messages/srv/detail/mesh__struct.h"


// Include directives for member types
// Member `mesh`
#include "shape_msgs/msg/mesh.h"
// Member `mesh`
#include "shape_msgs/msg/detail/mesh__rosidl_typesupport_introspection_c.h"
// Member `vertex_colors`
#include "std_msgs/msg/color_rgba.h"
// Member `vertex_colors`
#include "std_msgs/msg/detail/color_rgba__rosidl_typesupport_introspection_c.h"
// Member `pose`
#include "vision_msgs/msg/detection3_d_array.h"
// Member `pose`
#include "vision_msgs/msg/detail/detection3_d_array__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  opendr_ros2_messages__srv__Mesh_Response__init(message_memory);
}

void Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_fini_function(void * message_memory)
{
  opendr_ros2_messages__srv__Mesh_Response__fini(message_memory);
}

size_t Mesh_Response__rosidl_typesupport_introspection_c__size_function__ColorRGBA__vertex_colors(
  const void * untyped_member)
{
  const std_msgs__msg__ColorRGBA__Sequence * member =
    (const std_msgs__msg__ColorRGBA__Sequence *)(untyped_member);
  return member->size;
}

const void * Mesh_Response__rosidl_typesupport_introspection_c__get_const_function__ColorRGBA__vertex_colors(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__ColorRGBA__Sequence * member =
    (const std_msgs__msg__ColorRGBA__Sequence *)(untyped_member);
  return &member->data[index];
}

void * Mesh_Response__rosidl_typesupport_introspection_c__get_function__ColorRGBA__vertex_colors(
  void * untyped_member, size_t index)
{
  std_msgs__msg__ColorRGBA__Sequence * member =
    (std_msgs__msg__ColorRGBA__Sequence *)(untyped_member);
  return &member->data[index];
}

bool Mesh_Response__rosidl_typesupport_introspection_c__resize_function__ColorRGBA__vertex_colors(
  void * untyped_member, size_t size)
{
  std_msgs__msg__ColorRGBA__Sequence * member =
    (std_msgs__msg__ColorRGBA__Sequence *)(untyped_member);
  std_msgs__msg__ColorRGBA__Sequence__fini(member);
  return std_msgs__msg__ColorRGBA__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_member_array[3] = {
  {
    "mesh",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__srv__Mesh_Response, mesh),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vertex_colors",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__srv__Mesh_Response, vertex_colors),  // bytes offset in struct
    NULL,  // default value
    Mesh_Response__rosidl_typesupport_introspection_c__size_function__ColorRGBA__vertex_colors,  // size() function pointer
    Mesh_Response__rosidl_typesupport_introspection_c__get_const_function__ColorRGBA__vertex_colors,  // get_const(index) function pointer
    Mesh_Response__rosidl_typesupport_introspection_c__get_function__ColorRGBA__vertex_colors,  // get(index) function pointer
    Mesh_Response__rosidl_typesupport_introspection_c__resize_function__ColorRGBA__vertex_colors  // resize(index) function pointer
  },
  {
    "pose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages__srv__Mesh_Response, pose),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_members = {
  "opendr_ros2_messages__srv",  // message namespace
  "Mesh_Response",  // message name
  3,  // number of fields
  sizeof(opendr_ros2_messages__srv__Mesh_Response),
  Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_member_array,  // message members
  Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_type_support_handle = {
  0,
  &Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_opendr_ros2_messages
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, srv, Mesh_Response)() {
  Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, shape_msgs, msg, Mesh)();
  Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, ColorRGBA)();
  Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vision_msgs, msg, Detection3DArray)();
  if (!Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_type_support_handle.typesupport_identifier) {
    Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &Mesh_Response__rosidl_typesupport_introspection_c__Mesh_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "opendr_ros2_messages/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "opendr_ros2_messages/srv/detail/mesh__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_service_members = {
  "opendr_ros2_messages__srv",  // service namespace
  "Mesh",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_Request_message_type_support_handle,
  NULL  // response message
  // opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_Response_message_type_support_handle
};

static rosidl_service_type_support_t opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_service_type_support_handle = {
  0,
  &opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, srv, Mesh_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, srv, Mesh_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_opendr_ros2_messages
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, srv, Mesh)() {
  if (!opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_service_type_support_handle.typesupport_identifier) {
    opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, srv, Mesh_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, opendr_ros2_messages, srv, Mesh_Response)()->data;
  }

  return &opendr_ros2_messages__srv__detail__mesh__rosidl_typesupport_introspection_c__Mesh_service_type_support_handle;
}
