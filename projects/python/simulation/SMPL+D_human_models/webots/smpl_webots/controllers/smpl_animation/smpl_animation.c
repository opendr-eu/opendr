/*
 * Copyright 2020-2023 OpenDR European Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <webots/robot.h>
#include <webots/skin.h>
#include <webots/supervisor.h>
#include <smpl_util.h>
#include <quaternion_private.h>
#include <vector3_private.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "npy_array.h"
#include "unistd.h"
#define TIME_STEP 32

double *quaternion_mult(double *q, double *r){
    static double rn[4];
    rn[0] = r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3];
    rn[1] = r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2];
    rn[2] = r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1];
    rn[3] = r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0];
    return rn;
}

double *point_rotation_by_quaternion(double *p, double *q){
    double pa[4] = {0, p[0], p[1], p[2]};
    double q_conj[4] = {q[0], -1*q[1], -1*q[2], -1*q[3]};
    double *pn1;
    pn1 = quaternion_mult(q,pa);
    double *pn2;
    pn2 = quaternion_mult(pn1,q_conj);
    static double pf[3];
    pf[0] = pn2[1];
    pf[1] = pn2[2];
    pf[2] = pn2[3];
    return pf;
}

void euler_to_quat(double* euler_angle, WbuQuaternion* quat){
    WbuVector3 axes[3] = {wbu_vector3(X_AXIS), wbu_vector3(Y_AXIS), wbu_vector3(Z_AXIS)};
    WbuQuaternion q;
    *quat = wbu_quaternion_zero();
    for(int r=0;r<3;r++){
        q = wbu_quaternion_from_axis_angle(axes[r].x, axes[r].y, axes[r].z, euler_angle[r]);
        *quat = wbu_quaternion_multiply(q, *quat);
        *quat = wbu_quaternion_normalize(*quat);
    }
    return;
}

int main(int argc, char **argv) {
  int exit_args = 0;
  char *motion_file;
  char *skin_name;
  if (argc < 2){
     exit_args = 1;
  }
  else{
     skin_name = strtok(argv[1], " ");
     motion_file = strtok(NULL, " ");
  }
  if (exit_args==1){
     fprintf(stderr, "The Skin device name or the animation file is not specified in the controllerArgs field.\n");
     return 0;
  }
  wb_robot_init();
  WbDeviceTag skin = wb_robot_get_device(skin_name);
  char orient_path[100];
  char tr_path[100];;
  strcpy(orient_path, motion_file);
  strcat(orient_path, "/poses.npy" );
  strcpy(tr_path, motion_file);
  strcat(tr_path, "/trans.npy" );
  fprintf(stderr, "%s\n%s\n", orient_path, tr_path);
  if ( access( orient_path, F_OK ) != 0  ||  access( tr_path, F_OK ) != 0 ){
    fprintf(stderr, "The animation files do not exist.\n");
    return 0;
  }

  // Initialize model
  SmplSkel smplSkel;
  initialize_skel(&smplSkel);

  // read animation files
  read_smpl_orientation(orient_path, &smplSkel);
  read_smpl_translation(tr_path, &smplSkel);

  // Get the number of bones in the Skin device
  unsigned int skin_bone_count = wb_skin_get_bone_count(skin);

  if (skin_bone_count == 0) {
    printf("The Skin model has no bones to animate.\n");
    return 0;
  }

  bool is_position_offset_set = false;
  int cnt =0;
  double position_offset[3];
  WbuQuaternion skin_root_quat =  wbu_quaternion_zero();
  double trans[3] = {0,0,0};

  //Apply animation
  while (wb_robot_step(TIME_STEP) != -1){

    // Set initial position and orientation
    if (!is_position_offset_set) {
	const double *skin_root_position = wb_skin_get_bone_position(skin, 0, true);
	position_offset[0] = skin_root_position[0] - smplSkel.translation[cnt][0]*100;
	position_offset[1] = skin_root_position[1] - smplSkel.translation[cnt][1]*100;
	position_offset[2] = skin_root_position[2] - smplSkel.translation[cnt][2]*100;

        double rot[3] = {smplSkel.orientation[cnt][0], smplSkel.orientation[cnt][1], smplSkel.orientation[cnt][2]};
        double results[4];
        euler_to_quat(rot, &skin_root_quat);
        double skin_root_quat_norm = skin_root_quat.w*skin_root_quat.w + skin_root_quat.x*skin_root_quat.x + skin_root_quat.y*skin_root_quat.y + skin_root_quat.z*skin_root_quat.z;
        skin_root_quat.w = skin_root_quat.w/skin_root_quat_norm;
        skin_root_quat.x = -skin_root_quat.x/skin_root_quat_norm;
        skin_root_quat.y = -skin_root_quat.y/skin_root_quat_norm;
        skin_root_quat.z = -skin_root_quat.z/skin_root_quat_norm;
        is_position_offset_set = true;
    }
    // Compute current orientations of joints
    double orientations[25][4];
    for (int i = 1; i < skin_bone_count-1; ++i) {
       double rot[3] = {smplSkel.orientation[cnt][smplSkel.ids[i]*3], smplSkel.orientation[cnt][smplSkel.ids[i]*3 + 1], smplSkel.orientation[cnt][smplSkel.ids[i]*3 + 2]};
       double results[4];
       WbuQuaternion quat;
       euler_to_quat(rot, &quat);
       wbu_quaternion_to_axis_angle(quat, results);
       for(int j = 0; j < 4; j++){
          orientations[i][j] = results[j];
      }
    }

    trans[0] = position_offset[0] + smplSkel.translation[cnt][0]*100;
    trans[1] = position_offset[1] + smplSkel.translation[cnt][1]*100;
    trans[2] = position_offset[2] + smplSkel.translation[cnt][2]*100;

    for(int i=0;i<25;i++){
        //Apply joint orientations. Ignore orientation hands (ids=22,23), global translation (ids=24), global orientation (ids=0)
        if (smplSkel.ids[i]!=0  && smplSkel.ids[i]!=23 && smplSkel.ids[i]!=24 && smplSkel.ids[i]!=22){
            wb_skin_set_bone_orientation(skin, i, orientations[i], false);
        }

        //Apply global orientation
        if (smplSkel.ids[i]==0 ){
            double rot[3] = {smplSkel.orientation[cnt][0], smplSkel.orientation[cnt][1], smplSkel.orientation[cnt][2]};
            double orient_root[4];
            WbuQuaternion quat = wbu_quaternion_zero();
            euler_to_quat(rot, &quat);
            quat = wbu_quaternion_multiply(skin_root_quat, quat);
            wbu_quaternion_to_axis_angle(quat, orient_root);
            //wb_skin_set_bone_orientation(skin, i, orient_root, false);
        }

    }
    //wb_skin_set_bone_position(skin, 0, trans, false);
    cnt = cnt + 1;

    if(cnt>=smplSkel.numFrames){
     cnt = 0;
    }
  }

  wb_robot_cleanup();

  return 0;
}
