source /opt/ros/${ROS_DISTRO}/setup.bash
roscore &
sleep 5

TOOL=""
if [ "$#" -ge 1 ]; then
  TOOL=$1
fi

if [ $TOOL == "control/mobile_manipulation" ]; then
  echo "Lauch mobile manipulation specific nodes"
  source ${OPENDR_HOME}/projects/python/control/mobile_manipulation/mobile_manipulation_ws/devel/setup.bash
  roslaunch mobile_manipulation_rl pr2_analytical.launch &
fi