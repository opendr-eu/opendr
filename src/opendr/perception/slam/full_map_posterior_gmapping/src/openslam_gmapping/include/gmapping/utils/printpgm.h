
/*
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

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

ostream &printpgm(ostream &os, int xsize, int ysize, const double *const *matrix) {
  if (!os)
    return os;
  os << "P5" << endl << xsize << endl << ysize << endl << 255 << endl;
  for (int y = ysize - 1; y >= 0; y--) {
    for (int x = 0; x < xsize; x++) {
      unsigned char c = (unsigned char)(255 * fabs(1. - matrix[x][y]));
      os.put(c);
    }
  }
  return os;
}
