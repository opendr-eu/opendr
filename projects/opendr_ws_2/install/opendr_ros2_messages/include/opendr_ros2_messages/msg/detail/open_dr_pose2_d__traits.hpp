// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from opendr_ros2_messages:msg/OpenDRPose2D.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__TRAITS_HPP_
#define OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__TRAITS_HPP_

#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<opendr_ros2_messages::msg::OpenDRPose2D>()
{
  return "opendr_ros2_messages::msg::OpenDRPose2D";
}

template<>
inline const char * name<opendr_ros2_messages::msg::OpenDRPose2D>()
{
  return "opendr_ros2_messages/msg/OpenDRPose2D";
}

template<>
struct has_fixed_size<opendr_ros2_messages::msg::OpenDRPose2D>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<opendr_ros2_messages::msg::OpenDRPose2D>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<opendr_ros2_messages::msg::OpenDRPose2D>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__TRAITS_HPP_
