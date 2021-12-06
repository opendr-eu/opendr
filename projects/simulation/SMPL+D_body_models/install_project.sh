#!/bin/sh

cp -r ./webots $WEBOTS_HOME/projects
cd $WEBOTS_HOME/projects/smpl_webots/libraries/smpl_util
make
cd $WEBOTS_HOME/projects/smpl_webots/controllers/smpl_animation
make
mkdir $WEBOTS_HOME/projects/smpl_webots/skins/model-202
cp $OPENDR_HOME/projects/simulation/SMPL+D_body_models/fbx_models/female/model-202/model-202.fbx $WEBOTS_HOME/projects/smpl_webots/skins/model-202/model-202.fbx
mkdir $WEBOTS_HOME/projects/smpl_webots/protos/textures/model-202
cp $OPENDR_HOME/projects/simulation/SMPL+D_body_models/fbx_models/female/model-202/texture.png $WEBOTS_HOME/projects/smpl_webots/protos/textures/model-202/texture.png
