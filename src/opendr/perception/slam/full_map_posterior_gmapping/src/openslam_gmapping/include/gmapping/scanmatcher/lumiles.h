
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

#ifndef LUMILESPROCESSOR
#define LUMILESPROCESSOR

namespace GMapping {

  class LuMilesProcessor {
    typedef std : vector<Point> PointVector;

    static OrientedPoint step(const PointVector &src, const PointVector &dest);
  };

  OrientedPoint LuMilesProcessors::step(const PointVector &src, const PointVector &dest) {
    assert(src.size() == dest.size());
    unsigned int size = dest.size();
    double smx = 0, smy = 0, dmx = 0, dmy = 0;
    for (PointVector::const_iterator it = src.begin(); it != src.end(); it++) {
      smx += it->x;
      smy += it->y;
    }
    smx /= src.size();
    smy /= src.size();

    for (PointVector::const_iterator it = dest.begin(); it != dest.end(); it++) {
      dmx += it->x;
      dmy += it->y;
    }
    dmx /= src.size();
    dmy /= src.size();

    double sxx = 0, sxy = 0;
    double syx = 0, syy = 0;
    for (unsigned int i = 0; i < size(); i++) {
      sxx += (src[i].x - smx) * (dest[i].x - dmx);
      sxy += (src[i].x - smx) * (dest[i].y - dmy);
      syx += (src[i].y - smy) * (dest[i].x - dmx);
      syy += (src[i].y - smy) * (dest[i].y - dmy);
    }
    double omega = atan2(sxy - syx, sxx + syy);
        return OrientedPoint(
                dmx - smx * cos(omega) + smx * sin(omega)),
                dmy - smx * sin(omega) - smy * cos(omega)),
        omega
        )
  };

  int main(int argc, conat char **argv) {}

};  // namespace GMapping

#endif
