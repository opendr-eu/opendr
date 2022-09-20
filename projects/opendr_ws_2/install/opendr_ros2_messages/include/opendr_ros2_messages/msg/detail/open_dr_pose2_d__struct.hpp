// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from opendr_ros2_messages:msg/OpenDRPose2D.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__STRUCT_HPP_
#define OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'keypoint_list'
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__opendr_ros2_messages__msg__OpenDRPose2D __attribute__((deprecated))
#else
# define DEPRECATED__opendr_ros2_messages__msg__OpenDRPose2D __declspec(deprecated)
#endif

namespace opendr_ros2_messages
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct OpenDRPose2D_
{
  using Type = OpenDRPose2D_<ContainerAllocator>;

  explicit OpenDRPose2D_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->pose_id = 0l;
      this->conf = 0.0f;
    }
  }

  explicit OpenDRPose2D_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->pose_id = 0l;
      this->conf = 0.0f;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _pose_id_type =
    int32_t;
  _pose_id_type pose_id;
  using _conf_type =
    float;
  _conf_type conf;
  using _keypoint_list_type =
    std::vector<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>, typename ContainerAllocator::template rebind<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>>::other>;
  _keypoint_list_type keypoint_list;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__pose_id(
    const int32_t & _arg)
  {
    this->pose_id = _arg;
    return *this;
  }
  Type & set__conf(
    const float & _arg)
  {
    this->conf = _arg;
    return *this;
  }
  Type & set__keypoint_list(
    const std::vector<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>, typename ContainerAllocator::template rebind<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>>::other> & _arg)
  {
    this->keypoint_list = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator> *;
  using ConstRawPtr =
    const opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__opendr_ros2_messages__msg__OpenDRPose2D
    std::shared_ptr<opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__opendr_ros2_messages__msg__OpenDRPose2D
    std::shared_ptr<opendr_ros2_messages::msg::OpenDRPose2D_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const OpenDRPose2D_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->pose_id != other.pose_id) {
      return false;
    }
    if (this->conf != other.conf) {
      return false;
    }
    if (this->keypoint_list != other.keypoint_list) {
      return false;
    }
    return true;
  }
  bool operator!=(const OpenDRPose2D_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct OpenDRPose2D_

// alias to use template instance with default allocator
using OpenDRPose2D =
  opendr_ros2_messages::msg::OpenDRPose2D_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace opendr_ros2_messages

#endif  // OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D__STRUCT_HPP_
