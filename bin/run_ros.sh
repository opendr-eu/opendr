source /opt/ros/${ROS_DISTRO}/setup.bash
roscore &
sleep 5

TOOL=""
if [ "$#" -ge 1 ]; then
  TOOL=$1
fi

if [ $TOOL == "control/mobile_manipulation" ]
  cd projects/opendr_ws
  roslaunch mobile_manipulation_rl pr2_analytical.launch &
  cd ../..
fi