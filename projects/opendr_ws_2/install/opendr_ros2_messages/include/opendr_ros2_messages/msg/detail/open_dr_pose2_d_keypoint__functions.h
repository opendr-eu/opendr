// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from opendr_ros2_messages:msg/OpenDRPose2DKeypoint.idl
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__FUNCTIONS_H_
#define OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "opendr_ros2_messages/msg/rosidl_generator_c__visibility_control.h"

#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__struct.h"

/// Initialize msg/OpenDRPose2DKeypoint message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * opendr_ros2_messages__msg__OpenDRPose2DKeypoint
 * )) before or use
 * opendr_ros2_messages__msg__OpenDRPose2DKeypoint__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_opendr_ros2_messages
bool
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__init(opendr_ros2_messages__msg__OpenDRPose2DKeypoint * msg);

/// Finalize msg/OpenDRPose2DKeypoint message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_opendr_ros2_messages
void
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini(opendr_ros2_messages__msg__OpenDRPose2DKeypoint * msg);

/// Create msg/OpenDRPose2DKeypoint message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * opendr_ros2_messages__msg__OpenDRPose2DKeypoint__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_opendr_ros2_messages
opendr_ros2_messages__msg__OpenDRPose2DKeypoint *
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__create();

/// Destroy msg/OpenDRPose2DKeypoint message.
/**
 * It calls
 * opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_opendr_ros2_messages
void
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__destroy(opendr_ros2_messages__msg__OpenDRPose2DKeypoint * msg);


/// Initialize array of msg/OpenDRPose2DKeypoint messages.
/**
 * It allocates the memory for the number of elements and calls
 * opendr_ros2_messages__msg__OpenDRPose2DKeypoint__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_opendr_ros2_messages
bool
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__init(opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * array, size_t size);

/// Finalize array of msg/OpenDRPose2DKeypoint messages.
/**
 * It calls
 * opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_opendr_ros2_messages
void
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__fini(opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * array);

/// Create array of msg/OpenDRPose2DKeypoint messages.
/**
 * It allocates the memory for the array and calls
 * opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_opendr_ros2_messages
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence *
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__create(size_t size);

/// Destroy array of msg/OpenDRPose2DKeypoint messages.
/**
 * It calls
 * opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_opendr_ros2_messages
void
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__destroy(opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * array);

#ifdef __cplusplus
}
#endif

#endif  // OPENDR_ROS2_MESSAGES__MSG__DETAIL__OPEN_DR_POSE2_D_KEYPOINT__FUNCTIONS_H_
