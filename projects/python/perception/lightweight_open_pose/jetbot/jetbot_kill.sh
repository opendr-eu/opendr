#!bin/bash
echo 'Attemping to stop jetbot (this might take a while if the load is high)...'
python3 -c 'from jetbot import Robot;Robot().stop()'
