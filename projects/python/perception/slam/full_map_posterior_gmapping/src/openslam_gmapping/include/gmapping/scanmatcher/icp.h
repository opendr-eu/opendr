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

#ifndef _ICP_H_
#define _ICP_H_

#include <gmapping/utils/point.h>
#include <list>
#include <utility>
#include <vector>

namespace GMapping {
  typedef std::pair<Point, Point> PointPair;

  template<typename PointPairContainer> double icpStep(OrientedPoint &retval, const PointPairContainer &container) {
    typedef typename PointPairContainer::const_iterator ContainerIterator;
    PointPair mean = std::make_pair(Point(0., 0.), Point(0., 0.));
    int size = 0;
    for (ContainerIterator it = container.begin(); it != container.end(); it++) {
      mean.first = mean.first + it->first;
      mean.second = mean.second + it->second;
      size++;
    }
    mean.first = mean.first * (1. / size);
    mean.second = mean.second * (1. / size);
    double sxx = 0, sxy = 0, syx = 0, syy = 0;

    for (ContainerIterator it = container.begin(); it != container.end(); it++) {
      PointPair mf = std::make_pair(it->first - mean.first, it->second - mean.second);
      sxx += mf.first.x * mf.second.x;
      sxy += mf.first.x * mf.second.y;
      syx += mf.first.y * mf.second.x;
      syy += mf.first.y * mf.second.y;
    }
    retval.theta = atan2(sxy - syx, sxx + sxy);
    double s = sin(retval.theta), c = cos(retval.theta);
    retval.x = mean.second.x - (c * mean.first.x - s * mean.first.y);
    retval.y = mean.second.y - (s * mean.first.x + c * mean.first.y);

    double error = 0;
    for (ContainerIterator it = container.begin(); it != container.end(); it++) {
      Point delta(c * it->first.x - s * it->first.y + retval.x - it->second.x,
                  s * it->first.x + c * it->first.y + retval.y - it->second.y);
      error += delta * delta;
    }
    return error;
  }

  template<typename PointPairContainer> double icpNonlinearStep(OrientedPoint &retval, const PointPairContainer &container) {
    typedef typename PointPairContainer::const_iterator ContainerIterator;
    PointPair mean = std::make_pair(Point(0., 0.), Point(0., 0.));
    int size = 0;
    for (ContainerIterator it = container.begin(); it != container.end(); it++) {
      mean.first = mean.first + it->first;
      mean.second = mean.second + it->second;
      size++;
    }

    mean.first = mean.first * (1. / size);
    mean.second = mean.second * (1. / size);

    double ms = 0, mc = 0;
    for (ContainerIterator it = container.begin(); it != container.end(); it++) {
      PointPair mf = std::make_pair(it->first - mean.first, it->second - mean.second);
      double dalpha = atan2(mf.second.y, mf.second.x) - atan2(mf.first.y, mf.first.x);
      double gain = sqrt(mean.first * mean.first);
      ms += gain * sin(dalpha);
      mc += gain * cos(dalpha);
    }
    retval.theta = atan2(ms, mc);
    double s = sin(retval.theta), c = cos(retval.theta);
    retval.x = mean.second.x - (c * mean.first.x - s * mean.first.y);
    retval.y = mean.second.y - (s * mean.first.x + c * mean.first.y);

    double error = 0;
    for (ContainerIterator it = container.begin(); it != container.end(); it++) {
      Point delta(c * it->first.x - s * it->first.y + retval.x - it->second.x,
                  s * it->first.x + c * it->first.y + retval.y - it->second.y);
      error += delta * delta;
    }
    return error;
  }

}  // namespace GMapping

#endif
