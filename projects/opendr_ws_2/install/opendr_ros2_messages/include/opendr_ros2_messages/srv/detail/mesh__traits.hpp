// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from opendr_ros2_messages:srv/Mesh.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__TRAITS_HPP_
#define OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__TRAITS_HPP_

#include "opendr_ros2_messages/srv/detail/mesh__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'rgb_img'
// Member 'msk_img'
#include "sensor_msgs/msg/detail/image__traits.hpp"
// Member 'extract_pose'
#include "std_msgs/msg/detail/bool__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<opendr_ros2_messages::srv::Mesh_Request>()
{
  return "opendr_ros2_messages::srv::Mesh_Request";
}

template<>
inline const char * name<opendr_ros2_messages::srv::Mesh_Request>()
{
  return "opendr_ros2_messages/srv/Mesh_Request";
}

template<>
struct has_fixed_size<opendr_ros2_messages::srv::Mesh_Request>
  : std::integral_constant<bool, has_fixed_size<sensor_msgs::msg::Image>::value && has_fixed_size<std_msgs::msg::Bool>::value> {};

template<>
struct has_bounded_size<opendr_ros2_messages::srv::Mesh_Request>
  : std::integral_constant<bool, has_bounded_size<sensor_msgs::msg::Image>::value && has_bounded_size<std_msgs::msg::Bool>::value> {};

template<>
struct is_message<opendr_ros2_messages::srv::Mesh_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'mesh'
#include "shape_msgs/msg/detail/mesh__traits.hpp"
// Member 'pose'
#include "vision_msgs/msg/detail/detection3_d_array__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<opendr_ros2_messages::srv::Mesh_Response>()
{
  return "opendr_ros2_messages::srv::Mesh_Response";
}

template<>
inline const char * name<opendr_ros2_messages::srv::Mesh_Response>()
{
  return "opendr_ros2_messages/srv/Mesh_Response";
}

template<>
struct has_fixed_size<opendr_ros2_messages::srv::Mesh_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<opendr_ros2_messages::srv::Mesh_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<opendr_ros2_messages::srv::Mesh_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<opendr_ros2_messages::srv::Mesh>()
{
  return "opendr_ros2_messages::srv::Mesh";
}

template<>
inline const char * name<opendr_ros2_messages::srv::Mesh>()
{
  return "opendr_ros2_messages/srv/Mesh";
}

template<>
struct has_fixed_size<opendr_ros2_messages::srv::Mesh>
  : std::integral_constant<
    bool,
    has_fixed_size<opendr_ros2_messages::srv::Mesh_Request>::value &&
    has_fixed_size<opendr_ros2_messages::srv::Mesh_Response>::value
  >
{
};

template<>
struct has_bounded_size<opendr_ros2_messages::srv::Mesh>
  : std::integral_constant<
    bool,
    has_bounded_size<opendr_ros2_messages::srv::Mesh_Request>::value &&
    has_bounded_size<opendr_ros2_messages::srv::Mesh_Response>::value
  >
{
};

template<>
struct is_service<opendr_ros2_messages::srv::Mesh>
  : std::true_type
{
};

template<>
struct is_service_request<opendr_ros2_messages::srv::Mesh_Request>
  : std::true_type
{
};

template<>
struct is_service_response<opendr_ros2_messages::srv::Mesh_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__TRAITS_HPP_
