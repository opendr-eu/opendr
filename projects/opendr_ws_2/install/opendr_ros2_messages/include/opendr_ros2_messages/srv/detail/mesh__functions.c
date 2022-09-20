// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from opendr_ros2_messages:srv/Mesh.idl
// generated code does not contain a copyright notice
#include "opendr_ros2_messages/srv/detail/mesh__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// Include directives for member types
// Member `rgb_img`
// Member `msk_img`
#include "sensor_msgs/msg/detail/image__functions.h"
// Member `extract_pose`
#include "std_msgs/msg/detail/bool__functions.h"

bool
opendr_ros2_messages__srv__Mesh_Request__init(opendr_ros2_messages__srv__Mesh_Request * msg)
{
  if (!msg) {
    return false;
  }
  // rgb_img
  if (!sensor_msgs__msg__Image__init(&msg->rgb_img)) {
    opendr_ros2_messages__srv__Mesh_Request__fini(msg);
    return false;
  }
  // msk_img
  if (!sensor_msgs__msg__Image__init(&msg->msk_img)) {
    opendr_ros2_messages__srv__Mesh_Request__fini(msg);
    return false;
  }
  // extract_pose
  if (!std_msgs__msg__Bool__init(&msg->extract_pose)) {
    opendr_ros2_messages__srv__Mesh_Request__fini(msg);
    return false;
  }
  return true;
}

void
opendr_ros2_messages__srv__Mesh_Request__fini(opendr_ros2_messages__srv__Mesh_Request * msg)
{
  if (!msg) {
    return;
  }
  // rgb_img
  sensor_msgs__msg__Image__fini(&msg->rgb_img);
  // msk_img
  sensor_msgs__msg__Image__fini(&msg->msk_img);
  // extract_pose
  std_msgs__msg__Bool__fini(&msg->extract_pose);
}

opendr_ros2_messages__srv__Mesh_Request *
opendr_ros2_messages__srv__Mesh_Request__create()
{
  opendr_ros2_messages__srv__Mesh_Request * msg = (opendr_ros2_messages__srv__Mesh_Request *)malloc(sizeof(opendr_ros2_messages__srv__Mesh_Request));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(opendr_ros2_messages__srv__Mesh_Request));
  bool success = opendr_ros2_messages__srv__Mesh_Request__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
opendr_ros2_messages__srv__Mesh_Request__destroy(opendr_ros2_messages__srv__Mesh_Request * msg)
{
  if (msg) {
    opendr_ros2_messages__srv__Mesh_Request__fini(msg);
  }
  free(msg);
}


bool
opendr_ros2_messages__srv__Mesh_Request__Sequence__init(opendr_ros2_messages__srv__Mesh_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  opendr_ros2_messages__srv__Mesh_Request * data = NULL;
  if (size) {
    data = (opendr_ros2_messages__srv__Mesh_Request *)calloc(size, sizeof(opendr_ros2_messages__srv__Mesh_Request));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = opendr_ros2_messages__srv__Mesh_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        opendr_ros2_messages__srv__Mesh_Request__fini(&data[i - 1]);
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
opendr_ros2_messages__srv__Mesh_Request__Sequence__fini(opendr_ros2_messages__srv__Mesh_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      opendr_ros2_messages__srv__Mesh_Request__fini(&array->data[i]);
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

opendr_ros2_messages__srv__Mesh_Request__Sequence *
opendr_ros2_messages__srv__Mesh_Request__Sequence__create(size_t size)
{
  opendr_ros2_messages__srv__Mesh_Request__Sequence * array = (opendr_ros2_messages__srv__Mesh_Request__Sequence *)malloc(sizeof(opendr_ros2_messages__srv__Mesh_Request__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = opendr_ros2_messages__srv__Mesh_Request__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
opendr_ros2_messages__srv__Mesh_Request__Sequence__destroy(opendr_ros2_messages__srv__Mesh_Request__Sequence * array)
{
  if (array) {
    opendr_ros2_messages__srv__Mesh_Request__Sequence__fini(array);
  }
  free(array);
}


// Include directives for member types
// Member `mesh`
#include "shape_msgs/msg/detail/mesh__functions.h"
// Member `vertex_colors`
#include "std_msgs/msg/detail/color_rgba__functions.h"
// Member `pose`
#include "vision_msgs/msg/detail/detection3_d_array__functions.h"

bool
opendr_ros2_messages__srv__Mesh_Response__init(opendr_ros2_messages__srv__Mesh_Response * msg)
{
  if (!msg) {
    return false;
  }
  // mesh
  if (!shape_msgs__msg__Mesh__init(&msg->mesh)) {
    opendr_ros2_messages__srv__Mesh_Response__fini(msg);
    return false;
  }
  // vertex_colors
  if (!std_msgs__msg__ColorRGBA__Sequence__init(&msg->vertex_colors, 0)) {
    opendr_ros2_messages__srv__Mesh_Response__fini(msg);
    return false;
  }
  // pose
  if (!vision_msgs__msg__Detection3DArray__init(&msg->pose)) {
    opendr_ros2_messages__srv__Mesh_Response__fini(msg);
    return false;
  }
  return true;
}

void
opendr_ros2_messages__srv__Mesh_Response__fini(opendr_ros2_messages__srv__Mesh_Response * msg)
{
  if (!msg) {
    return;
  }
  // mesh
  shape_msgs__msg__Mesh__fini(&msg->mesh);
  // vertex_colors
  std_msgs__msg__ColorRGBA__Sequence__fini(&msg->vertex_colors);
  // pose
  vision_msgs__msg__Detection3DArray__fini(&msg->pose);
}

opendr_ros2_messages__srv__Mesh_Response *
opendr_ros2_messages__srv__Mesh_Response__create()
{
  opendr_ros2_messages__srv__Mesh_Response * msg = (opendr_ros2_messages__srv__Mesh_Response *)malloc(sizeof(opendr_ros2_messages__srv__Mesh_Response));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(opendr_ros2_messages__srv__Mesh_Response));
  bool success = opendr_ros2_messages__srv__Mesh_Response__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
opendr_ros2_messages__srv__Mesh_Response__destroy(opendr_ros2_messages__srv__Mesh_Response * msg)
{
  if (msg) {
    opendr_ros2_messages__srv__Mesh_Response__fini(msg);
  }
  free(msg);
}


bool
opendr_ros2_messages__srv__Mesh_Response__Sequence__init(opendr_ros2_messages__srv__Mesh_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  opendr_ros2_messages__srv__Mesh_Response * data = NULL;
  if (size) {
    data = (opendr_ros2_messages__srv__Mesh_Response *)calloc(size, sizeof(opendr_ros2_messages__srv__Mesh_Response));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = opendr_ros2_messages__srv__Mesh_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        opendr_ros2_messages__srv__Mesh_Response__fini(&data[i - 1]);
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
opendr_ros2_messages__srv__Mesh_Response__Sequence__fini(opendr_ros2_messages__srv__Mesh_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      opendr_ros2_messages__srv__Mesh_Response__fini(&array->data[i]);
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

opendr_ros2_messages__srv__Mesh_Response__Sequence *
opendr_ros2_messages__srv__Mesh_Response__Sequence__create(size_t size)
{
  opendr_ros2_messages__srv__Mesh_Response__Sequence * array = (opendr_ros2_messages__srv__Mesh_Response__Sequence *)malloc(sizeof(opendr_ros2_messages__srv__Mesh_Response__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = opendr_ros2_messages__srv__Mesh_Response__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
opendr_ros2_messages__srv__Mesh_Response__Sequence__destroy(opendr_ros2_messages__srv__Mesh_Response__Sequence * array)
{
  if (array) {
    opendr_ros2_messages__srv__Mesh_Response__Sequence__fini(array);
  }
  free(array);
}
