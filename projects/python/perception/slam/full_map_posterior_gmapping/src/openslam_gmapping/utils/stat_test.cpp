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

#include "gmapping/utils/stat.h"
#include <math.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace GMapping;

// struct Covariance3{
// 	double xx, yy, tt, xy, xt, yt;
// };

#define SAMPLES_NUMBER 10000

int main(int argc, char **argv) {
  Covariance3 cov = {1., 0.01, 0.01, 0, 0, 0};
  EigenCovariance3 ecov(cov);
  cout << "EigenValues: " << ecov.eval[0] << " " << ecov.eval[1] << " " << ecov.eval[2] << endl;

  cout << "EigenVectors:" << endl;
  cout << ecov.evec[0][0] << " " << ecov.evec[0][1] << " " << ecov.evec[0][2] << endl;
  cout << ecov.evec[1][0] << " " << ecov.evec[1][1] << " " << ecov.evec[1][2] << endl;
  cout << ecov.evec[2][0] << " " << ecov.evec[2][1] << " " << ecov.evec[2][2] << endl;

  EigenCovariance3 rcov(ecov.rotate(M_PI / 4));
  cout << "*************** Rotated ***************" << endl;
  cout << "EigenValues: " << rcov.eval[0] << " " << rcov.eval[1] << " " << rcov.eval[2] << endl;

  cout << "EigenVectors:" << endl;
  cout << rcov.evec[0][0] << " " << rcov.evec[0][1] << " " << rcov.evec[0][2] << endl;
  cout << rcov.evec[1][0] << " " << rcov.evec[1][1] << " " << rcov.evec[1][2] << endl;
  cout << rcov.evec[2][0] << " " << rcov.evec[2][1] << " " << rcov.evec[2][2] << endl;

  cout << "sampling:" << endl;
  ofstream fs("stat_test.dat");
  std::vector<OrientedPoint> points;
  for (unsigned int i = 0; i < SAMPLES_NUMBER; i++) {
    OrientedPoint op = rcov.sample();
    points.push_back(op);
    fs << op.x << " " << op.y << " " << op.theta << endl;
  }
  fs.close();
  std::vector<OrientedPoint>::iterator b = points.begin();
  std::vector<OrientedPoint>::iterator e = points.end();
  Gaussian3 gaussian = computeGaussianFromSamples(b, e);
  cov = gaussian.cov;
  ecov = gaussian.covariance;
  cout << "*************** Estimated with Templates ***************" << endl;
  cout << "EigenValues: " << ecov.eval[0] << " " << ecov.eval[1] << " " << ecov.eval[2] << endl;
  cout << "EigenVectors:" << endl;
  cout << ecov.evec[0][0] << " " << ecov.evec[0][1] << " " << ecov.evec[0][2] << endl;
  cout << ecov.evec[1][0] << " " << ecov.evec[1][1] << " " << ecov.evec[1][2] << endl;
  cout << ecov.evec[2][0] << " " << ecov.evec[2][1] << " " << ecov.evec[2][2] << endl;
  gaussian.computeFromSamples(points);
  ecov = gaussian.covariance;
  cout << "*************** Estimated without Templates ***************" << endl;
  cout << "EigenValues: " << ecov.eval[0] << " " << ecov.eval[1] << " " << ecov.eval[2] << endl;
  cout << "EigenVectors:" << endl;
  cout << ecov.evec[0][0] << " " << ecov.evec[0][1] << " " << ecov.evec[0][2] << endl;
  cout << ecov.evec[1][0] << " " << ecov.evec[1][1] << " " << ecov.evec[1][2] << endl;
  cout << ecov.evec[2][0] << " " << ecov.evec[2][1] << " " << ecov.evec[2][2] << endl;
}
