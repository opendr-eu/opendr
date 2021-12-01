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

#include <webots/robot.h>
#include <webots/skin.h>
#include <webots/supervisor.h>
#include <webots/smpl_util.h>
#include <webots/bvh_util.h>
#include "quaternion_private.h"
#include "vector3_private.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "npy_array.h"
#include "unistd.h"
#define TIME_STEP 32


int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "The Skin device name or the animation file is not specified in the controllerArgs field.\n");
    return 0;
  }

  wb_robot_init();
  WbDeviceTag skin = wb_robot_get_device(argv[1]);
  char *oriet_path = strcat( argv[2], "/poses.npy" )
  char *tr_path = = strcat( argv[2], "/trans.npy" )
  if ( access( oriet_path, F_OK ) != 0  ||  access( tr_path, F_OK ) != 0 ){
    fprintf(stderr, "The animation files do not exist.\n");
    return 0;
  }
  SmplSkel smplSkel;
  initialize_skel(&smplSkel);
  int counter;
  

  read_smpl_orientation(oriet_path, &smplSkel);
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
  bool init_rot = false;
  double in_rot[3];
  WbuQuaternion in_quat;
  double trans[3];
  trans[0] = 0;
  trans[1] = 0;
  trans[2] = 0;
  while (wb_robot_step(TIME_STEP) != -1){
    double orientations[25][4];
    for (int i = 0; i < skin_bone_count; ++i) {
      WbuQuaternion frame_rotation = wbu_quaternion_zero();
      double result[4];
      double rot[3];
      WbuQuaternion q;
      WbuVector3 axes[3];
      axes[0] = wbu_vector3(X_AXIS);
      axes[1] = wbu_vector3(Y_AXIS);
      axes[2] = wbu_vector3(Z_AXIS);
      result[0] = 0;
      result[1] = 0;
      result[2] = 0;
      result[3] = 0;
      rot[0] = 0;
      rot[1] = 0;
      rot[2] = 0;
      int s;
      int joint_id;
      for (s=0;s<25;s++){
         if (smplSkel.ids[i] == s){
            joint_id = s;
         }
      }
      rot[0] = smplSkel.orientation[cnt][joint_id*3];
      rot[1] = smplSkel.orientation[cnt][joint_id*3 +1];
      rot[2] = smplSkel.orientation[cnt][joint_id*3 +2];
      //if (joint_id == 0){
      //  rot[0] = rot[0];
      //  rot[1] = rot[1];
      //  rot[2] = rot[2];
      //}
      fprintf(stderr, "%f %f %f %f\n", rot[0], rot[1], rot[2], rot[3]);
      int r = 0;
      for(r=0;r<3;r++){
            q = wbu_quaternion_from_axis_angle(axes[r].x, axes[r].y, axes[r].z, rot[r]);
            int j = 0;
            for (j=0; j < 3; ++j) {
               if (j != r)
                  axes[j] = wbu_vector3_rotate_by_quaternion(axes[j], q);
            }
            frame_rotation = wbu_quaternion_multiply(q, frame_rotation);
            frame_rotation = wbu_quaternion_normalize(frame_rotation);
      }
      wbu_quaternion_to_axis_angle(frame_rotation,result);
      orientations[i][0] = result[0];
      orientations[i][1] = result[1];
      orientations[i][2] = result[2];
      orientations[i][3] = result[3];
    }
    if (!is_position_offset_set) {
	const double *skin_root_position = wb_skin_get_bone_position(skin, 1, true);
	position_offset[0] = skin_root_position[0] + smplSkel.translation[cnt][0]*100;
	position_offset[2] = skin_root_position[2] + smplSkel.translation[cnt][1]*100;
	position_offset[1] = skin_root_position[1] + smplSkel.translation[cnt][2]*100;
	is_position_offset_set = true;

    }
    int i = 0;

    trans[0] = position_offset[0] -smplSkel.translation[cnt][0]*100;
    trans[2] = position_offset[2] -smplSkel.translation[cnt][1]*100;
    trans[1] = position_offset[1] -smplSkel.translation[cnt][2]*100;
    
    for(i=0;i<25;i++){
           if(i==19 || i == 24){
           orientations[i][0] = 0; 
           orientations[i][1] = 0;
           orientations[i][2] = 0;
           orientations[i][3] = 0;
           }
           else if (i>1){
                wb_skin_set_bone_orientation(skin, i, orientations[i], false);
           }
    }
    double f_trans[4];
    cnt = cnt + 1;
  }



  wb_robot_cleanup();

  return 0;
}
