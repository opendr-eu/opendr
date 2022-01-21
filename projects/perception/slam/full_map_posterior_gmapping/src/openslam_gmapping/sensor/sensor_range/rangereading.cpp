// Copyright 2020-2022 OpenDR European Project
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

#include "gmapping/sensor/sensor_range/rangereading.h"
#include <assert.h>
#include <gmapping/utils/gvalues.h>
#include <sys/types.h>
#include <iostream>
#include <limits>

namespace GMapping {

  using namespace std;

  RangeReading::RangeReading(const RangeSensor *rs, double time) : SensorReading(rs, time) {}

  RangeReading::RangeReading(unsigned int n_beams, const double *d, const RangeSensor *rs, double time) :
    SensorReading(rs, time) {
    assert(n_beams == rs->beams().size());
    resize(n_beams);
    for (unsigned int i = 0; i < size(); i++)
      (*this)[i] = d[i];
  }

  RangeReading::~RangeReading() {
  }

  unsigned int RangeReading::rawView(double *v, double density) const {
    if (density == 0) {
      for (unsigned int i = 0; i < size(); i++)
        v[i] = (*this)[i];
    } else {
      Point lastPoint(0, 0);
      uint suppressed = 0;
      for (unsigned int i = 0; i < size(); i++) {
        const RangeSensor *rs = dynamic_cast<const RangeSensor *>(getSensor());
        assert(rs);
        Point lp(cos(rs->beams()[i].pose.theta) * (*this)[i], sin(rs->beams()[i].pose.theta) * (*this)[i]);
        Point dp = lastPoint - lp;
        double distance = sqrt(dp * dp);
        if (distance < density) {
          //				v[i]=MAXDOUBLE;
          v[i] = std::numeric_limits<double>::max();
          suppressed++;
        } else {
          lastPoint = lp;
          v[i] = (*this)[i];
        }
        // std::cerr<< __PRETTY_FUNCTION__ << std::endl;
        // std::cerr<< "suppressed " << suppressed <<"/"<<size() << std::endl;
      }
    }
    //	return size();
    return static_cast<unsigned int>(size());
  };

  unsigned int RangeReading::activeBeams(double density) const {
    if (density == 0.)
      return size();
    int ab = 0;
    Point lastPoint(0, 0);
    uint suppressed = 0;
    for (unsigned int i = 0; i < size(); i++) {
      const RangeSensor *rs = dynamic_cast<const RangeSensor *>(getSensor());
      assert(rs);
      Point lp(cos(rs->beams()[i].pose.theta) * (*this)[i], sin(rs->beams()[i].pose.theta) * (*this)[i]);
      Point dp = lastPoint - lp;
      double distance = sqrt(dp * dp);
      if (distance < density) {
        suppressed++;
      } else {
        lastPoint = lp;
        ab++;
      }
      // std::cerr<< __PRETTY_FUNCTION__ << std::endl;
      // std::cerr<< "suppressed " << suppressed <<"/"<<size() << std::endl;
    }
    return ab;
  }

  std::vector<Point> RangeReading::cartesianForm(double maxRange) const {
    const RangeSensor *rangeSensor = dynamic_cast<const RangeSensor *>(getSensor());
    assert(rangeSensor && rangeSensor->beams().size());
    //	uint m_beams=rangeSensor->beams().size();
    uint m_beams = static_cast<unsigned int>(rangeSensor->beams().size());
    std::vector<Point> cartesianPoints(m_beams);
    double px, py, ps, pc;
    px = rangeSensor->getPose().x;
    py = rangeSensor->getPose().y;
    ps = sin(rangeSensor->getPose().theta);
    pc = cos(rangeSensor->getPose().theta);
    for (unsigned int i = 0; i < m_beams; i++) {
      const double &rho = (*this)[i];
      const double &s = rangeSensor->beams()[i].s;
      const double &c = rangeSensor->beams()[i].c;
      if (rho >= maxRange) {
        cartesianPoints[i] = Point(0, 0);
      } else {
        Point p = Point(rangeSensor->beams()[i].pose.x + c * rho, rangeSensor->beams()[i].pose.y + s * rho);
        cartesianPoints[i].x = px + pc * p.x - ps * p.y;
        cartesianPoints[i].y = py + ps * p.x + pc * p.y;
      }
    }
    return cartesianPoints;
  }

};  // namespace GMapping
