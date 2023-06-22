source ${OPENDR_HOME}/projects/python/control/mobile_manipulation/mobile_manipulation_ws/devel/setup.bash
roscore &
sleep 5
roslaunch mobile_manipulation_rl pr2_analytical.launch &
