#!/bin/bash

# Please sets the parths appropriately to your Webotsinstallation and according to your python version
export OPENDR_PATH="opendr_internal/src/"
export PYTHONPATH="/usr/local/webots/lib/controller/python38:$OPENDR_PATH"
export LD_LIBRARY_PATH="/usr/local/webots/lib/controller"


python3 fall_controller.py --setup 1 --opendr --local --active
#python3 fall_controller.py --video --setup 0 --local
#python3 fall_controller.py --video --setup 0 --opendr --local


#python3 fall_controller.py --opendr --video --setup 1 --local
#python3 fall_controller.py --opendr --video --setup 1 --active --local


#python3 fall_controller.py --opendr --setup 2 --local
#python3 fall_controller.py --opendr --active --setup 2 --local
