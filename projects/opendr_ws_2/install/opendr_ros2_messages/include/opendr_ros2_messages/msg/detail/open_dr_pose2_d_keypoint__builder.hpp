// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from opendr_ros2_messages:msg/OpenDRPose2DKeypoint.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__BUILDER_HPP_
#define OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__BUILDER_HPP_

#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace opendr_ros2_messages
{

namespace msg
{

namespace builder
{

class Init_OpenDRPose2DKeypoint_y
{
public:
  explicit Init_OpenDRPose2DKeypoint_y(::opendr_ros2_messages::msg::OpenDRPose2DKeypoint & msg)
  : msg_(msg)
  {}
  ::opendr_ros2_messages::msg::OpenDRPose2DKeypoint y(::opendr_ros2_messages::msg::OpenDRPose2DKeypoint::_y_type arg)
  {
    msg_.y = std::move(arg);
    return std::move(msg_);
  }

private:
  ::opendr_ros2_messages::msg::OpenDRPose2DKeypoint msg_;
};

class Init_OpenDRPose2DKeypoint_x
{
public:
  explicit Init_OpenDRPose2DKeypoint_x(::opendr_ros2_messages::msg::OpenDRPose2DKeypoint & msg)
  : msg_(msg)
  {}
  Init_OpenDRPose2DKeypoint_y x(::opendr_ros2_messages::msg::OpenDRPose2DKeypoint::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_OpenDRPose2DKeypoint_y(msg_);
  }

private:
  ::opendr_ros2_messages::msg::OpenDRPose2DKeypoint msg_;
};

class Init_OpenDRPose2DKeypoint_kpt_name
{
public:
  Init_OpenDRPose2DKeypoint_kpt_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_OpenDRPose2DKeypoint_x kpt_name(::opendr_ros2_messages::msg::OpenDRPose2DKeypoint::_kpt_name_type arg)
  {
    msg_.kpt_name = std::move(arg);
    return Init_OpenDRPose2DKeypoint_x(msg_);
  }

private:
  ::opendr_ros2_messages::msg::OpenDRPose2DKeypoint msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::opendr_ros2_messages::msg::OpenDRPose2DKeypoint>()
{
  return opendr_ros2_messages::msg::builder::Init_OpenDRPose2DKeypoint_kpt_name();
}

}  // namespace opendr_ros2_messages

#endif  // OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__BUILDER_HPP_
