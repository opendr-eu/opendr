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

#include "smpl_util.h"
#include "quaternion_private.h"
#include "vector3_private.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void initialize_skel(SmplSkel *smplSkel){
  strcpy(smplSkel->joints[0], "Root");
  smplSkel->ids[0] = 24;
  strcpy(smplSkel->joints[1], "Pelvis");
  smplSkel->ids[1] = 0;
  strcpy(smplSkel->joints[2], "L_Hip");
  smplSkel->ids[2] = 1;
  strcpy(smplSkel->joints[3], "L_Knee");
  smplSkel->ids[3] = 4;
  strcpy(smplSkel->joints[4], "L_Ankle");
  smplSkel->ids[4] = 7;
  strcpy(smplSkel->joints[5], "L_Foot");
  smplSkel->ids[5] = 10;
  strcpy(smplSkel->joints[6], "R_Hip");
  smplSkel->ids[6] = 2;
  strcpy(smplSkel->joints[7], "R_Knee");
  smplSkel->ids[7] = 5;
  strcpy(smplSkel->joints[8], "R_Ankle");
  smplSkel->ids[8] = 8;
  strcpy(smplSkel->joints[9], "R_Foot");
  smplSkel->ids[9] = 11;
  strcpy(smplSkel->joints[10], "Spine1");
  smplSkel->ids[10] = 3;
  strcpy(smplSkel->joints[11], "Spine2");
  smplSkel->ids[11] = 6;
  strcpy(smplSkel->joints[12], "Spine3");
  smplSkel->ids[12] = 9;
  strcpy(smplSkel->joints[13], "Neck");
  smplSkel->ids[13] = 12;
  strcpy(smplSkel->joints[14], "Head");
  smplSkel->ids[14] = 15;
  strcpy(smplSkel->joints[15], "L_Collar");
  smplSkel->ids[15] = 13;
  strcpy(smplSkel->joints[16], "L_Shoulder");
  smplSkel->ids[16] = 16;
  strcpy(smplSkel->joints[17], "L_Elbow");
  smplSkel->ids[17] = 18;
  strcpy(smplSkel->joints[18], "L_Wrist");
  smplSkel->ids[18] = 20;
  strcpy(smplSkel->joints[19], "L_Hand");
  smplSkel->ids[19] = 22;
  strcpy(smplSkel->joints[20], "R_Collar");
  smplSkel->ids[20] = 14;
  strcpy(smplSkel->joints[21], "R_Shoulder");
  smplSkel->ids[21] = 17;
  strcpy(smplSkel->joints[22], "R_Elbow");
  smplSkel->ids[22] = 19;
  strcpy(smplSkel->joints[23], "R_Wrist");
  smplSkel->ids[23] = 21;
  strcpy(smplSkel->joints[24], "R_Hand");
  smplSkel->ids[24] = 23;
}

void read_smpl_orientation(const char *filename_orient, SmplSkel *smplSkel){
    FILE *f;
    f = fopen(filename_orient,"r");
    char b[1000];
    char V;
    unsigned short int HEADER_LEN;
    int HL_bytes = 2;
    // skip first 6 bytes. NUMPY
    fread(b, 6, 1, f);
    // read major version byte
    fread(&V, 1, 1, f);
    //skip minor version byte
    fread(b, 1, 1, f);
    if (V == 2) {
      HL_bytes = 4; // 4 bytes if major version is true
     }
    fread(&HEADER_LEN, HL_bytes, 1, f);
    // skip the header
    fread(b, HEADER_LEN, 1, f);
    int counter = 0;
    const int N = 156;
    double *dummy_v = (double *) malloc((N) * sizeof(double ));
    while( fread(dummy_v, N*sizeof(double), 1, f)){
	counter = counter + 1;
    }
    fclose(f);
    smplSkel->numFrames = counter;
    f = fopen(filename_orient,"r");
    fread(b, 6, 1, f);
    fread(&V, 1, 1, f);
    fread(b, 1, 1, f);
    if (V == 2) {
      HL_bytes = 4;
     }
    fread(&HEADER_LEN, HL_bytes, 1, f);
    fread(b, HEADER_LEN, 1, f);

    // read the data
    smplSkel->orientation = (double **) malloc((smplSkel->numFrames) * sizeof(double *));
    for (int i=0;i<smplSkel->numFrames;i++){
       smplSkel->orientation[i] = (double *) malloc((N) * sizeof(double ));
       fread(smplSkel->orientation[i], N*sizeof(double), 1, f);
    }
}


void read_smpl_translation(const char *filename_transl, SmplSkel *smplSkel){
    FILE *f;
    f = fopen(filename_transl,"r");
    char b[1000];
    char V;
    unsigned short int HEADER_LEN;
    int HL_bytes = 2;
    // skip first 6 bytes. NUMPY
    fread(b, 6, 1, f);
    // read major version byte
    fread(&V, 1, 1, f);
    //skip minor version byte
    fread(b, 1, 1, f);
    if (V == 2) {
      HL_bytes = 4; // 4 bytes if major version is true
     }
    fread(&HEADER_LEN, HL_bytes, 1, f);
    // skip the header
    fread(b, HEADER_LEN, 1, f);
    int counter = 0;
    const int N = 3;
    double *dummy_v = (double *) malloc((N) * sizeof(double ));
    while( fread(dummy_v, N*sizeof(double), 1, f)){
	counter = counter + 1;
    }
    fclose(f);
    smplSkel->numFrames = counter;
    f = fopen(filename_transl,"r");
    fread(b, 6, 1, f);
    fread(&V, 1, 1, f);
    fread(b, 1, 1, f);
    if (V == 2) {
      HL_bytes = 4;
     }
    fread(&HEADER_LEN, HL_bytes, 1, f);
    fread(b, HEADER_LEN, 1, f);

    // read the data
    smplSkel->translation = (double **) malloc((smplSkel->numFrames) * sizeof(double *));
    for (int i=0;i<smplSkel->numFrames;i++){
       smplSkel->translation[i] = (double *) malloc((N) * sizeof(double ));
       fread(smplSkel->translation[i], N*sizeof(double), 1, f);
    }
}
