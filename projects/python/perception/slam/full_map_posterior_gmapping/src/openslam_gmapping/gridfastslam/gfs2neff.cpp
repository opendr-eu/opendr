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

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "usage gfs2neff <infilename> <nefffilename>" << endl;
    return -1;
  }
  ifstream is(argv[1]);
  if (!is) {
    cout << "could read file " << endl;
    return -1;
  }
  ofstream os(argv[2]);
  if (!os) {
    cout << "could write file " << endl;
    return -1;
  }
  unsigned int frame = 0;
  double neff = 0;
  while (is) {
    char buf[8192];
    is.getline(buf, 8192);
    istringstream lineStream(buf);
    string recordType;
    lineStream >> recordType;
    if (recordType == "FRAME") {
      lineStream >> frame;
    }
    if (recordType == "NEFF") {
      lineStream >> neff;
      os << frame << " " << neff << endl;
    }
  }
  os.close();
}
