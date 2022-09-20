// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from opendr_ros2_messages:msg/OpenDRPose2D.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__BUILDER_HPP_
#define OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__BUILDER_HPP_

#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace opendr_ros2_messages
{

namespace msg
{

namespace builder
{

class Init_OpenDRPose2D_keypoint_list
{
public:
  explicit Init_OpenDRPose2D_keypoint_list(::opendr_ros2_messages::msg::OpenDRPose2D & msg)
  : msg_(msg)
  {}
  ::opendr_ros2_messages::msg::OpenDRPose2D keypoint_list(::opendr_ros2_messages::msg::OpenDRPose2D::_keypoint_list_type arg)
  {
    msg_.keypoint_list = std::move(arg);
    return std::move(msg_);
  }

private:
  ::opendr_ros2_messages::msg::OpenDRPose2D msg_;
};

class Init_OpenDRPose2D_conf
{
public:
  explicit Init_OpenDRPose2D_conf(::opendr_ros2_messages::msg::OpenDRPose2D & msg)
  : msg_(msg)
  {}
  Init_OpenDRPose2D_keypoint_list conf(::opendr_ros2_messages::msg::OpenDRPose2D::_conf_type arg)
  {
    msg_.conf = std::move(arg);
    return Init_OpenDRPose2D_keypoint_list(msg_);
  }

private:
  ::opendr_ros2_messages::msg::OpenDRPose2D msg_;
};

class Init_OpenDRPose2D_pose_id
{
public:
  explicit Init_OpenDRPose2D_pose_id(::opendr_ros2_messages::msg::OpenDRPose2D & msg)
  : msg_(msg)
  {}
  Init_OpenDRPose2D_conf pose_id(::opendr_ros2_messages::msg::OpenDRPose2D::_pose_id_type arg)
  {
    msg_.pose_id = std::move(arg);
    return Init_OpenDRPose2D_conf(msg_);
  }

private:
  ::opendr_ros2_messages::msg::OpenDRPose2D msg_;
};

class Init_OpenDRPose2D_header
{
public:
  Init_OpenDRPose2D_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_OpenDRPose2D_pose_id header(::opendr_ros2_messages::msg::OpenDRPose2D::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_OpenDRPose2D_pose_id(msg_);
  }

private:
  ::opendr_ros2_messages::msg::OpenDRPose2D msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::opendr_ros2_messages::msg::OpenDRPose2D>()
{
  return opendr_ros2_messages::msg::builder::Init_OpenDRPose2D_header();
}

}  // namespace opendr_ros2_messages

#endif  // OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__BUILDER_HPP_
