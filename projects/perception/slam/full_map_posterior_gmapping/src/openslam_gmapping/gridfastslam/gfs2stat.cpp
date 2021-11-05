
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

#include <gmapping/particlefilter/particlefilter.h>
#include <gmapping/utils/stat.h>
#include <fstream>
#include <iostream>
#include "gmapping/gridfastslam/gfsreader.h"

using namespace std;
using namespace GMapping;
using namespace GMapping::GFSReader;

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "usage gfs2stat <infilename> <outfilename>" << endl;
    return 0;
  }
  ifstream is(argv[1]);
  if (!is) {
    cout << "no file found: " << argv[1] << endl;
    return 0;
  }
  ofstream os(argv[2]);
  if (!os) {
    cout << "cannot open file: " << argv[1] << endl;
    return 0;
  }
  cout << "loading... " << flush;
  RecordList rl;
  rl.read(is);
  cout << " done" << endl;
  int count = -1;
  for (RecordList::const_iterator it = rl.begin(); it != rl.end(); it++) {
    count++;
    const ScanMatchRecord *rec = dynamic_cast<const ScanMatchRecord *>(*it);
    if (!rec)
      continue;
    Gaussian3 gaussian;
    /*
    vector<double> nweights;
    cout << "N"<< flush;
    back_insert_iterator< vector<double> > out(nweights);
    toNormalForm(out,rec->weights.begin(), rec->weights.end());
    cout << "G"<< flush;
    gaussian.computeFromSamples(rec->poses, nweights);
    */
    gaussian.computeFromSamples(rec->poses);
    cout << "E" << flush;
    os << count << " ";
    os << gaussian.mean.x << " ";
    os << gaussian.mean.y << " ";
    os << gaussian.mean.theta << " ";
    os << gaussian.covariance.eval[0] << " ";
    os << gaussian.covariance.eval[1] << " ";
    os << gaussian.covariance.eval[2] << endl;
  }
  os.close();
}
