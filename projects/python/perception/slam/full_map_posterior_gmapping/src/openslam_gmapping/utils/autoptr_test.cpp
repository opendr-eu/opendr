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

#include "gmapping/utils/autoptr.h"
#include <iostream>

using namespace std;
using namespace GMapping;

typedef autoptr<double> DoubleAutoPtr;

int main(int argc, const char *const *argv) {
  double *d1 = new double(10.);
  double *d2 = new double(20.);
  cout << "Construction test" << endl;
  DoubleAutoPtr pd1(d1);
  DoubleAutoPtr pd2(d2);
  cout << *pd1 << " " << *pd2 << endl;
  cout << "Copy Construction" << endl;
  DoubleAutoPtr pd3(pd1);
  cout << *pd3 << endl;
  cout << "assignment" << endl;
  pd3 = pd2;
  pd1 = pd2;
  cout << *pd1 << " " << *pd2 << " " << *pd3 << " " << endl;
  cout << "conversion operator" << endl;
  DoubleAutoPtr nullPtr;
  cout << "conversion operator " << !(nullPtr) << endl;
  cout << "neg conversion operator " << nullPtr << endl;
  cout << "conversion operator " << (int)pd1 << endl;
  cout << "neg conversion operator " << !(pd1) << endl;
}
