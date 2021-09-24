source ${OPENDR_HOME}/lib/catkin_ws_mobile_manipulation/devel/setup.bash
roscore &
sleep 5
roslaunch mobile_manipulation_rl pr2_analytical.launch &