// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from opendr_ros2_messages:srv/Mesh.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__BUILDER_HPP_
#define OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__BUILDER_HPP_

#include "opendr_ros2_messages/srv/detail/mesh__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace opendr_ros2_messages
{

namespace srv
{

namespace builder
{

class Init_Mesh_Request_extract_pose
{
public:
  explicit Init_Mesh_Request_extract_pose(::opendr_ros2_messages::srv::Mesh_Request & msg)
  : msg_(msg)
  {}
  ::opendr_ros2_messages::srv::Mesh_Request extract_pose(::opendr_ros2_messages::srv::Mesh_Request::_extract_pose_type arg)
  {
    msg_.extract_pose = std::move(arg);
    return std::move(msg_);
  }

private:
  ::opendr_ros2_messages::srv::Mesh_Request msg_;
};

class Init_Mesh_Request_msk_img
{
public:
  explicit Init_Mesh_Request_msk_img(::opendr_ros2_messages::srv::Mesh_Request & msg)
  : msg_(msg)
  {}
  Init_Mesh_Request_extract_pose msk_img(::opendr_ros2_messages::srv::Mesh_Request::_msk_img_type arg)
  {
    msg_.msk_img = std::move(arg);
    return Init_Mesh_Request_extract_pose(msg_);
  }

private:
  ::opendr_ros2_messages::srv::Mesh_Request msg_;
};

class Init_Mesh_Request_rgb_img
{
public:
  Init_Mesh_Request_rgb_img()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Mesh_Request_msk_img rgb_img(::opendr_ros2_messages::srv::Mesh_Request::_rgb_img_type arg)
  {
    msg_.rgb_img = std::move(arg);
    return Init_Mesh_Request_msk_img(msg_);
  }

private:
  ::opendr_ros2_messages::srv::Mesh_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::opendr_ros2_messages::srv::Mesh_Request>()
{
  return opendr_ros2_messages::srv::builder::Init_Mesh_Request_rgb_img();
}

}  // namespace opendr_ros2_messages


namespace opendr_ros2_messages
{

namespace srv
{

namespace builder
{

class Init_Mesh_Response_pose
{
public:
  explicit Init_Mesh_Response_pose(::opendr_ros2_messages::srv::Mesh_Response & msg)
  : msg_(msg)
  {}
  ::opendr_ros2_messages::srv::Mesh_Response pose(::opendr_ros2_messages::srv::Mesh_Response::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return std::move(msg_);
  }

private:
  ::opendr_ros2_messages::srv::Mesh_Response msg_;
};

class Init_Mesh_Response_vertex_colors
{
public:
  explicit Init_Mesh_Response_vertex_colors(::opendr_ros2_messages::srv::Mesh_Response & msg)
  : msg_(msg)
  {}
  Init_Mesh_Response_pose vertex_colors(::opendr_ros2_messages::srv::Mesh_Response::_vertex_colors_type arg)
  {
    msg_.vertex_colors = std::move(arg);
    return Init_Mesh_Response_pose(msg_);
  }

private:
  ::opendr_ros2_messages::srv::Mesh_Response msg_;
};

class Init_Mesh_Response_mesh
{
public:
  Init_Mesh_Response_mesh()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Mesh_Response_vertex_colors mesh(::opendr_ros2_messages::srv::Mesh_Response::_mesh_type arg)
  {
    msg_.mesh = std::move(arg);
    return Init_Mesh_Response_vertex_colors(msg_);
  }

private:
  ::opendr_ros2_messages::srv::Mesh_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::opendr_ros2_messages::srv::Mesh_Response>()
{
  return opendr_ros2_messages::srv::builder::Init_Mesh_Response_mesh();
}

}  // namespace opendr_ros2_messages

#endif  // OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__BUILDER_HPP_
