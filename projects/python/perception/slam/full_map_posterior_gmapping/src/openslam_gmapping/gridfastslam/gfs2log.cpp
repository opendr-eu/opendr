// Copyright 2020-2023 OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gmapping/utils/point.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>
#include "gmapping/gridfastslam/gfsreader.h"

#define MAX_LINE_LENGHT (1000000)

using namespace std;
using namespace GMapping;
using namespace GMapping::GFSReader;

int main(int argc, const char *const *argv) {
  if (argc < 3) {
    cout << "usage gfs2log [-err] [-neff] [-part] [-odom] <infilename> <outfilename>" << endl;
    cout << "  -odom : dump raw odometry in ODOM message instead of inpolated corrected one" << endl;
    return -1;
  }
  bool err = 0;
  bool neff = 0;
  bool part = 0;
  bool odom = 0;
  //	int particle_num;
  unsigned int c = 1;
  if (!strcmp(argv[c], "-err")) {
    err = true;
    c++;
  }
  if (!strcmp(argv[c], "-neff")) {
    neff = true;
    c++;
  }
  if (!strcmp(argv[c], "-part")) {
    part = true;
    c++;
  }
  if (!strcmp(argv[c], "-odom")) {
    odom = true;
    c++;
  }
  ifstream is(argv[c]);
  if (!is) {
    cout << "could read file " << endl;
    return -1;
  }
  c++;
  RecordList rl;
  rl.read(is);
  unsigned int bestidx = rl.getBestIdx();
  cout << endl << "best index = " << bestidx << endl;
  ofstream os(argv[c]);
  if (!os) {
    cout << "could write file " << endl;
    return -1;
  }
  rl.printPath(os, bestidx, err, odom);
  if (part)
    rl.printLastParticles(os);
  os.close();
  return 0;
}
