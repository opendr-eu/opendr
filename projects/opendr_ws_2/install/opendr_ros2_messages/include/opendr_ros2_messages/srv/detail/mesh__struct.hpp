// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from opendr_ros2_messages:srv/Mesh.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__STRUCT_HPP_
#define OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'rgb_img'
// Member 'msk_img'
#include "sensor_msgs/msg/detail/image__struct.hpp"
// Member 'extract_pose'
#include "std_msgs/msg/detail/bool__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__opendr_ros2_messages__srv__Mesh_Request __attribute__((deprecated))
#else
# define DEPRECATED__opendr_ros2_messages__srv__Mesh_Request __declspec(deprecated)
#endif

namespace opendr_ros2_messages
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Mesh_Request_
{
  using Type = Mesh_Request_<ContainerAllocator>;

  explicit Mesh_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : rgb_img(_init),
    msk_img(_init),
    extract_pose(_init)
  {
    (void)_init;
  }

  explicit Mesh_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : rgb_img(_alloc, _init),
    msk_img(_alloc, _init),
    extract_pose(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _rgb_img_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _rgb_img_type rgb_img;
  using _msk_img_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _msk_img_type msk_img;
  using _extract_pose_type =
    std_msgs::msg::Bool_<ContainerAllocator>;
  _extract_pose_type extract_pose;

  // setters for named parameter idiom
  Type & set__rgb_img(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->rgb_img = _arg;
    return *this;
  }
  Type & set__msk_img(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->msk_img = _arg;
    return *this;
  }
  Type & set__extract_pose(
    const std_msgs::msg::Bool_<ContainerAllocator> & _arg)
  {
    this->extract_pose = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__opendr_ros2_messages__srv__Mesh_Request
    std::shared_ptr<opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__opendr_ros2_messages__srv__Mesh_Request
    std::shared_ptr<opendr_ros2_messages::srv::Mesh_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Mesh_Request_ & other) const
  {
    if (this->rgb_img != other.rgb_img) {
      return false;
    }
    if (this->msk_img != other.msk_img) {
      return false;
    }
    if (this->extract_pose != other.extract_pose) {
      return false;
    }
    return true;
  }
  bool operator!=(const Mesh_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Mesh_Request_

// alias to use template instance with default allocator
using Mesh_Request =
  opendr_ros2_messages::srv::Mesh_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace opendr_ros2_messages


// Include directives for member types
// Member 'mesh'
#include "shape_msgs/msg/detail/mesh__struct.hpp"
// Member 'vertex_colors'
#include "std_msgs/msg/detail/color_rgba__struct.hpp"
// Member 'pose'
#include "vision_msgs/msg/detail/detection3_d_array__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__opendr_ros2_messages__srv__Mesh_Response __attribute__((deprecated))
#else
# define DEPRECATED__opendr_ros2_messages__srv__Mesh_Response __declspec(deprecated)
#endif

namespace opendr_ros2_messages
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Mesh_Response_
{
  using Type = Mesh_Response_<ContainerAllocator>;

  explicit Mesh_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mesh(_init),
    pose(_init)
  {
    (void)_init;
  }

  explicit Mesh_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mesh(_alloc, _init),
    pose(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _mesh_type =
    shape_msgs::msg::Mesh_<ContainerAllocator>;
  _mesh_type mesh;
  using _vertex_colors_type =
    std::vector<std_msgs::msg::ColorRGBA_<ContainerAllocator>, typename ContainerAllocator::template rebind<std_msgs::msg::ColorRGBA_<ContainerAllocator>>::other>;
  _vertex_colors_type vertex_colors;
  using _pose_type =
    vision_msgs::msg::Detection3DArray_<ContainerAllocator>;
  _pose_type pose;

  // setters for named parameter idiom
  Type & set__mesh(
    const shape_msgs::msg::Mesh_<ContainerAllocator> & _arg)
  {
    this->mesh = _arg;
    return *this;
  }
  Type & set__vertex_colors(
    const std::vector<std_msgs::msg::ColorRGBA_<ContainerAllocator>, typename ContainerAllocator::template rebind<std_msgs::msg::ColorRGBA_<ContainerAllocator>>::other> & _arg)
  {
    this->vertex_colors = _arg;
    return *this;
  }
  Type & set__pose(
    const vision_msgs::msg::Detection3DArray_<ContainerAllocator> & _arg)
  {
    this->pose = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__opendr_ros2_messages__srv__Mesh_Response
    std::shared_ptr<opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__opendr_ros2_messages__srv__Mesh_Response
    std::shared_ptr<opendr_ros2_messages::srv::Mesh_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Mesh_Response_ & other) const
  {
    if (this->mesh != other.mesh) {
      return false;
    }
    if (this->vertex_colors != other.vertex_colors) {
      return false;
    }
    if (this->pose != other.pose) {
      return false;
    }
    return true;
  }
  bool operator!=(const Mesh_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Mesh_Response_

// alias to use template instance with default allocator
using Mesh_Response =
  opendr_ros2_messages::srv::Mesh_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace opendr_ros2_messages

namespace opendr_ros2_messages
{

namespace srv
{

struct Mesh
{
  using Request = opendr_ros2_messages::srv::Mesh_Request;
  using Response = opendr_ros2_messages::srv::Mesh_Response;
};

}  // namespace srv

}  // namespace opendr_ros2_messages

#endif  // OPENDR_ROS2_MESSAGES__SRV__DETAIL__MESH__STRUCT_HPP_
