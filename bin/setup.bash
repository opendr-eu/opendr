#!/bin/sh
export OPENDR_HOME=$PWD
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3

case $(lsb_release -r |cut -f2) in
  "18.04")
    export ROS_DISTRO=melodic;;
  "20.04")
    export ROS_DISTRO=noetic;;
  *)
    echo "Not tested for this ubuntu version" && exit 1;;
esac

if [[ ! -d "venv" ]]; then
	pip3 install virtualenv
	virtualenv -p python3 venv
fi
source venv/bin/activate
