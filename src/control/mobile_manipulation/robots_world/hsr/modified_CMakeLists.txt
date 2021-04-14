set(MOVEIT_PLUGIN_LIB_NAME hsrb_moveit_controller_manager)

include_directories(BEFORE
  thirdparty/moveit_simple_controller_manager/include
)

add_library(${MOVEIT_PLUGIN_LIB_NAME}
  thirdparty/moveit_simple_controller_manager/src/moveit_simple_controller_manager.cpp
  thirdparty/moveit_fake_controller_manager/src/moveit_fake_controller_manager.cpp
  thirdparty/moveit_fake_controller_manager/src/moveit_fake_controllers.cpp
)

target_link_libraries(${MOVEIT_PLUGIN_LIB_NAME}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)

install(TARGETS ${MOVEIT_PLUGIN_LIB_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

