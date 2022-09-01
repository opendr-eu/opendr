#!/bin/sh

cp -r ./webots/smpl_webots/ $WEBOTS_HOME/projects
cd $WEBOTS_HOME/projects/smpl_webots/libraries/smpl_util
make
cd $WEBOTS_HOME/projects/smpl_webots/controllers/smpl_animation
make
mkdir $WEBOTS_HOME/projects/smpl_webots/skins
mkdir $WEBOTS_HOME/projects/smpl_webots/skins/model-204
cp $OPENDR_HOME/projects/python/simulation/SMPL+D_human_models/fbx_models/female/204_0/204_0.fbx $WEBOTS_HOME/projects/smpl_webots/skins/model-204/model-204.fbx
mkdir $WEBOTS_HOME/projects/smpl_webots/protos/textures
mkdir $WEBOTS_HOME/projects/smpl_webots/protos/textures/model-204
cp $OPENDR_HOME/projects/python/simulation/SMPL+D_human_models/fbx_models/female/204_0/texture.jpg $WEBOTS_HOME/projects/smpl_webots/protos/textures/model-204/texture.jpg
