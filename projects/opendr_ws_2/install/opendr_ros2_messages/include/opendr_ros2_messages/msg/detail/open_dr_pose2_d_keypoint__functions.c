// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from opendr_ros2_messages:msg/OpenDRPose2DKeypoint.idl
// generated code does not contain a copyright notice
#include "opendr_ros2_messages/msg/detail/open_dr_pose2_d_keypoint__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


// Include directives for member types
// Member `kpt_name`
#include "rosidl_runtime_c/string_functions.h"

bool
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__init(opendr_ros2_messages__msg__OpenDRPose2DKeypoint * msg)
{
  if (!msg) {
    return false;
  }
  // kpt_name
  if (!rosidl_runtime_c__String__init(&msg->kpt_name)) {
    opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini(msg);
    return false;
  }
  // x
  // y
  return true;
}

void
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini(opendr_ros2_messages__msg__OpenDRPose2DKeypoint * msg)
{
  if (!msg) {
    return;
  }
  // kpt_name
  rosidl_runtime_c__String__fini(&msg->kpt_name);
  // x
  // y
}

opendr_ros2_messages__msg__OpenDRPose2DKeypoint *
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__create()
{
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint * msg = (opendr_ros2_messages__msg__OpenDRPose2DKeypoint *)malloc(sizeof(opendr_ros2_messages__msg__OpenDRPose2DKeypoint));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(opendr_ros2_messages__msg__OpenDRPose2DKeypoint));
  bool success = opendr_ros2_messages__msg__OpenDRPose2DKeypoint__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__destroy(opendr_ros2_messages__msg__OpenDRPose2DKeypoint * msg)
{
  if (msg) {
    opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini(msg);
  }
  free(msg);
}


bool
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__init(opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint * data = NULL;
  if (size) {
    data = (opendr_ros2_messages__msg__OpenDRPose2DKeypoint *)calloc(size, sizeof(opendr_ros2_messages__msg__OpenDRPose2DKeypoint));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = opendr_ros2_messages__msg__OpenDRPose2DKeypoint__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini(&data[i - 1]);
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
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__fini(opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      opendr_ros2_messages__msg__OpenDRPose2DKeypoint__fini(&array->data[i]);
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

opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence *
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__create(size_t size)
{
  opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * array = (opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence *)malloc(sizeof(opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__destroy(opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence * array)
{
  if (array) {
    opendr_ros2_messages__msg__OpenDRPose2DKeypoint__Sequence__fini(array);
  }
  free(array);
}
