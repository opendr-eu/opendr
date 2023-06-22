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

#include <stdbool.h>

typedef struct SmplSkel {
  int ids[25];
  char joints[25][100];
  double duration;
  double **orientation;
  double **translation;
  int numFrames;
}SmplSkel;

void initialize_skel(SmplSkel *smplSkel);
void read_smpl_orientation(const char *filename_orient, SmplSkel *smplSkel);
void read_smpl_translation(const char *filename_trans, SmplSkel *smplSkel);
