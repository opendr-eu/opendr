// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from opendr_ros2_messages:msg/OpenDRPose2D.idl
// generated code does not contain a copyright notice
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `keypoint_list`
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__functions.h"

bool
opendr_ros2_messages__msg__OpenDRPose2D__init(opendr_ros2_messages__msg__OpenDRPose2D * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    opendr_ros2_messages__msg__OpenDRPose2D__fini(msg);
    return false;
  }
  // pose_id
  // conf
  // keypoint_list
  if (!opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__init(&msg->keypoint_list, 0)) {
    opendr_ros2_messages__msg__OpenDRPose2D__fini(msg);
    return false;
  }
  return true;
}

void
opendr_ros2_messages__msg__OpenDRPose2D__fini(opendr_ros2_messages__msg__OpenDRPose2D * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // pose_id
  // conf
  // keypoint_list
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__fini(&msg->keypoint_list);
}

opendr_ros2_messages__msg__OpenDRPose2D *
opendr_ros2_messages__msg__OpenDRPose2D__create()
{
  opendr_ros2_messages__msg__OpenDRPose2D * msg = (opendr_ros2_messages__msg__OpenDRPose2D *)malloc(sizeof(opendr_ros2_messages__msg__OpenDRPose2D));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(opendr_ros2_messages__msg__OpenDRPose2D));
  bool success = opendr_ros2_messages__msg__OpenDRPose2D__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
opendr_ros2_messages__msg__OpenDRPose2D__destroy(opendr_ros2_messages__msg__OpenDRPose2D * msg)
{
  if (msg) {
    opendr_ros2_messages__msg__OpenDRPose2D__fini(msg);
  }
  free(msg);
}


bool
opendr_ros2_messages__msg__OpenDRPose2D__Sequence__init(opendr_ros2_messages__msg__OpenDRPose2D__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  opendr_ros2_messages__msg__OpenDRPose2D * data = NULL;
  if (size) {
    data = (opendr_ros2_messages__msg__OpenDRPose2D *)calloc(size, sizeof(opendr_ros2_messages__msg__OpenDRPose2D));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = opendr_ros2_messages__msg__OpenDRPose2D__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        opendr_ros2_messages__msg__OpenDRPose2D__fini(&data[i - 1]);
      }
      free(data);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
opendr_ros2_messages__msg__OpenDRPose2D__Sequence__fini(opendr_ros2_messages__msg__OpenDRPose2D__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      opendr_ros2_messages__msg__OpenDRPose2D__fini(&array->data[i]);
    }
    free(array->data);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

opendr_ros2_messages__msg__OpenDRPose2D__Sequence *
opendr_ros2_messages__msg__OpenDRPose2D__Sequence__create(size_t size)
{
  opendr_ros2_messages__msg__OpenDRPose2D__Sequence * array = (opendr_ros2_messages__msg__OpenDRPose2D__Sequence *)malloc(sizeof(opendr_ros2_messages__msg__OpenDRPose2D__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = opendr_ros2_messages__msg__OpenDRPose2D__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
opendr_ros2_messages__msg__OpenDRPose2D__Sequence__destroy(opendr_ros2_messages__msg__OpenDRPose2D__Sequence * array)
{
  if (array) {
    opendr_ros2_messages__msg__OpenDRPose2D__Sequence__fini(array);
  }
  free(array);
}
