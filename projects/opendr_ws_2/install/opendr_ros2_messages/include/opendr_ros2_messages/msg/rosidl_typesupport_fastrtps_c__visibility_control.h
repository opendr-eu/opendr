// generated from
// rosidl_typesupport_fastrtps_c/resource/rosidl_typesupport_fastrtps_c__visibility_control.h.in
// generated code does not contain a copyright notice

#ifndef OPENDR_ROS2_MESSAGES__MSG__ROSIDL_TYPESUPPORT_FASTRTPS_C__VISIBILITY_CONTROL_H_
#define OPENDR_ROS2_MESSAGES__MSG__ROSIDL_TYPESUPPORT_FASTRTPS_C__VISIBILITY_CONTROL_H_

#if __cplusplus
extern "C"
{
#endif

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define ROSIDL_TYPESUPPORT_FASTRTPS_C_EXPORT_opendr_ros2_messages __attribute__ ((dllexport))
    #define ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_opendr_ros2_messages __attribute__ ((dllimport))
  #else
    #define ROSIDL_TYPESUPPORT_FASTRTPS_C_EXPORT_opendr_ros2_messages __declspec(dllexport)
    #define ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_opendr_ros2_messages __declspec(dllimport)
  #endif
  #ifdef ROSIDL_TYPESUPPORT_FASTRTPS_C_BUILDING_DLL_opendr_ros2_messages
    #define ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_opendr_ros2_messages ROSIDL_TYPESUPPORT_FASTRTPS_C_EXPORT_opendr_ros2_messages
  #else
    #define ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_opendr_ros2_messages ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_opendr_ros2_messages
  #endif
#else
  #define ROSIDL_TYPESUPPORT_FASTRTPS_C_EXPORT_opendr_ros2_messages __attribute__ ((visibility("default")))
  #define ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_opendr_ros2_messages
  #if __GNUC__ >= 4
    #define ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_opendr_ros2_messages __attribute__ ((visibility("default")))
  #else
    #define ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_opendr_ros2_messages
  #endif
#endif

#if __cplusplus
}
#endif

#endif  // OPENDR_ROS2_MESSAGES__MSG__ROSIDL_TYPESUPPORT_FASTRTPS_C__VISIBILITY_CONTROL_H_
