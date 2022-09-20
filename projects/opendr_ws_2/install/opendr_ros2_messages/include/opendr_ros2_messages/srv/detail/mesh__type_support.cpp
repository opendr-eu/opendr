// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from opendr_ros2_messages:srv/Mesh.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "opendr_ros2_messages/srv/detail/mesh__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace opendr_ros2_messages
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void Mesh_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) opendr_ros2_messages::srv::Mesh_Request(_init);
}

void Mesh_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<opendr_ros2_messages::srv::Mesh_Request *>(message_memory);
  typed_message->~Mesh_Request();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Mesh_Request_message_member_array[3] = {
  {
    "rgb_img",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<sensor_msgs::msg::Image>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::srv::Mesh_Request, rgb_img),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "msk_img",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<sensor_msgs::msg::Image>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::srv::Mesh_Request, msk_img),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "extract_pose",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Bool>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::srv::Mesh_Request, extract_pose),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Mesh_Request_message_members = {
  "opendr_ros2_messages::srv",  // message namespace
  "Mesh_Request",  // message name
  3,  // number of fields
  sizeof(opendr_ros2_messages::srv::Mesh_Request),
  Mesh_Request_message_member_array,  // message members
  Mesh_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  Mesh_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Mesh_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Mesh_Request_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace opendr_ros2_messages


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<opendr_ros2_messages::srv::Mesh_Request>()
{
  return &::opendr_ros2_messages::srv::rosidl_typesupport_introspection_cpp::Mesh_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, opendr_ros2_messages, srv, Mesh_Request)() {
  return &::opendr_ros2_messages::srv::rosidl_typesupport_introspection_cpp::Mesh_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "array"
// already included above
// #include "cstddef"
// already included above
// #include "string"
// already included above
// #include "vector"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "opendr_ros2_messages/srv/detail/mesh__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/field_types.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace opendr_ros2_messages
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void Mesh_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) opendr_ros2_messages::srv::Mesh_Response(_init);
}

void Mesh_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<opendr_ros2_messages::srv::Mesh_Response *>(message_memory);
  typed_message->~Mesh_Response();
}

size_t size_function__Mesh_Response__vertex_colors(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<std_msgs::msg::ColorRGBA> *>(untyped_member);
  return member->size();
}

const void * get_const_function__Mesh_Response__vertex_colors(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<std_msgs::msg::ColorRGBA> *>(untyped_member);
  return &member[index];
}

void * get_function__Mesh_Response__vertex_colors(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<std_msgs::msg::ColorRGBA> *>(untyped_member);
  return &member[index];
}

void resize_function__Mesh_Response__vertex_colors(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<std_msgs::msg::ColorRGBA> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Mesh_Response_message_member_array[3] = {
  {
    "mesh",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<shape_msgs::msg::Mesh>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::srv::Mesh_Response, mesh),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "vertex_colors",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::ColorRGBA>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::srv::Mesh_Response, vertex_colors),  // bytes offset in struct
    nullptr,  // default value
    size_function__Mesh_Response__vertex_colors,  // size() function pointer
    get_const_function__Mesh_Response__vertex_colors,  // get_const(index) function pointer
    get_function__Mesh_Response__vertex_colors,  // get(index) function pointer
    resize_function__Mesh_Response__vertex_colors  // resize(index) function pointer
  },
  {
    "pose",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<vision_msgs::msg::Detection3DArray>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(opendr_ros2_messages::srv::Mesh_Response, pose),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Mesh_Response_message_members = {
  "opendr_ros2_messages::srv",  // message namespace
  "Mesh_Response",  // message name
  3,  // number of fields
  sizeof(opendr_ros2_messages::srv::Mesh_Response),
  Mesh_Response_message_member_array,  // message members
  Mesh_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  Mesh_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Mesh_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Mesh_Response_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace opendr_ros2_messages


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<opendr_ros2_messages::srv::Mesh_Response>()
{
  return &::opendr_ros2_messages::srv::rosidl_typesupport_introspection_cpp::Mesh_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, opendr_ros2_messages, srv, Mesh_Response)() {
  return &::opendr_ros2_messages::srv::rosidl_typesupport_introspection_cpp::Mesh_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"
// already included above
// #include "opendr_ros2_messages/srv/detail/mesh__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/service_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/service_type_support_decl.hpp"

namespace opendr_ros2_messages
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

// this is intentionally not const to allow initialization later to prevent an initialization race
static ::rosidl_typesupport_introspection_cpp::ServiceMembers Mesh_service_members = {
  "opendr_ros2_messages::srv",  // service namespace
  "Mesh",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<opendr_ros2_messages::srv::Mesh>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t Mesh_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Mesh_service_members,
  get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace opendr_ros2_messages


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<opendr_ros2_messages::srv::Mesh>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::opendr_ros2_messages::srv::rosidl_typesupport_introspection_cpp::Mesh_service_type_support_handle;
  // get a non-const and properly typed version of the data void *
  auto service_members = const_cast<::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
    static_cast<const ::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
      service_type_support->data));
  // make sure that both the request_members_ and the response_members_ are initialized
  // if they are not, initialize them
  if (
    service_members->request_members_ == nullptr ||
    service_members->response_members_ == nullptr)
  {
    // initialize the request_members_ with the static function from the external library
    service_members->request_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::opendr_ros2_messages::srv::Mesh_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::opendr_ros2_messages::srv::Mesh_Response
      >()->data
      );
  }
  // finally return the properly initialized service_type_support handle
  return service_type_support;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, opendr_ros2_messages, srv, Mesh)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<opendr_ros2_messages::srv::Mesh>();
}

#ifdef __cplusplus
}
#endif
