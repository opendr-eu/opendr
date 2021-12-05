#!/bin/sh

cp -r ./webots $WEBOTS_HOME/projects
cd $WEBOTS_HOME/projects/smpl_webots/libraries/smpl_util
make
cd $WEBOTS_HOME/projects/smpl_webots/controllers/smpl_animation
make
