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

#ifndef SMMAP_H
#define SMMAP_H

#include <gmapping/grid/harray2d.h>
#include <gmapping/grid/map.h>
#include <gmapping/utils/point.h>

#define SIGHT_INC 1

namespace GMapping {

  struct PointAccumulator {
    typedef point<float> FloatPoint;
    /* before
    PointAccumulator(int i=-1): acc(0,0), n(0), visits(0){assert(i==-1);}
    */
    /*after begin*/
    PointAccumulator() : acc(0, 0), n(0), visits(0), R(0) {}

    PointAccumulator(int i) : acc(0, 0), n(0), visits(0), R(0) { assert(i == -1); }

    /*after end*/
    inline void reset_inc();

    inline void update(bool value, const Point &p = Point(0, 0), double r = 0);

    inline Point mean() const { return 1. / n * Point(acc.x, acc.y); }

    inline operator double() const { return visits ? (double)n * SIGHT_INC / (double)visits : -1; }

    inline void add(const PointAccumulator &p) {
      acc = acc + p.acc;
      n += p.n;
      visits += p.visits;
      R += p.R;
    }

    static const PointAccumulator &Unknown();

    static PointAccumulator *unknown_ptr;
    FloatPoint acc;
    double R;       // Total distance that all rays travel in the cell during mapping
    int n, visits;  // Hits, Visits = Hits + Misses

    inline double entropy() const;
  };

  void PointAccumulator::update(bool value, const Point &p, double r) {
    if (value) {
      acc.x += static_cast<float>(p.x);
      acc.y += static_cast<float>(p.y);
      n++;
      visits += SIGHT_INC;
    } else
      visits++;

    R += r;
  }

  double PointAccumulator::entropy() const {
    if (!visits)
      return -log(.5);
    if (n == visits || n == 0)
      return 0;
    double x = (double)n * SIGHT_INC / (double)visits;
    return -(x * log(x) + (1 - x) * log(1 - x));
  }

  typedef Map<PointAccumulator, HierarchicalArray2D<PointAccumulator>> ScanMatcherMap;

};  // namespace GMapping

#endif
