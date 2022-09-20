// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from opendr_ros2_messages:msg/OpenDRPose2DKeypoint.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__STRUCT_HPP_
#define OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__opendr_ros2_messages__msg__OpenDRPose2DKeypoint __attribute__((deprecated))
#else
# define DEPRECATED__opendr_ros2_messages__msg__OpenDRPose2DKeypoint __declspec(deprecated)
#endif

namespace opendr_ros2_messages
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct OpenDRPose2DKeypoint_
{
  using Type = OpenDRPose2DKeypoint_<ContainerAllocator>;

  explicit OpenDRPose2DKeypoint_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->kpt_name = "";
      this->x = 0l;
      this->y = 0l;
    }
  }

  explicit OpenDRPose2DKeypoint_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : kpt_name(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->kpt_name = "";
      this->x = 0l;
      this->y = 0l;
    }
  }

  // field types and members
  using _kpt_name_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _kpt_name_type kpt_name;
  using _x_type =
    int32_t;
  _x_type x;
  using _y_type =
    int32_t;
  _y_type y;

  // setters for named parameter idiom
  Type & set__kpt_name(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->kpt_name = _arg;
    return *this;
  }
  Type & set__x(
    const int32_t & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const int32_t & _arg)
  {
    this->y = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator> *;
  using ConstRawPtr =
    const opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__opendr_ros2_messages__msg__OpenDRPose2DKeypoint
    std::shared_ptr<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__opendr_ros2_messages__msg__OpenDRPose2DKeypoint
    std::shared_ptr<opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const OpenDRPose2DKeypoint_ & other) const
  {
    if (this->kpt_name != other.kpt_name) {
      return false;
    }
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    return true;
  }
  bool operator!=(const OpenDRPose2DKeypoint_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct OpenDRPose2DKeypoint_

// alias to use template instance with default allocator
using OpenDRPose2DKeypoint =
  opendr_ros2_messages::msg::OpenDRPose2DKeypoint_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace opendr_ros2_messages

#endif  // OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__STRUCT_HPP_
