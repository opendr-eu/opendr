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

#define BOOST_TEST_MODULE FMP_GMapping_Test
#define BOOST_TEST_DYN_LINK
#define BOOST_AUTO_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include "gmapping/scanmatcher/gridlinetraversal.h"
#include "gmapping/utils/autoptr.h"

using namespace GMapping;

typedef autoptr<double> DoubleAutoPtr;

BOOST_AUTO_TEST_CASE(AutoPointerTest) {
  double *d1 = new double(10.);
  double *d2 = new double(20.);

  // Construction test
  DoubleAutoPtr pd1(d1);
  DoubleAutoPtr pd2(d2);
  BOOST_CHECK(*d1 == *pd1);
  BOOST_CHECK(*d2 == *pd2);

  // Copy Construction
  DoubleAutoPtr pd3(pd1);
  BOOST_CHECK(*pd3 == *pd1);

  // Assignment
  pd3 = pd2;
  pd1 = pd2;
  BOOST_CHECK(*pd1 == *pd2 && *pd2 == *pd3);
}

int bresenham_num_points(int x0, int y0, int x1, int y1, bool supercover) {
  IntPoint *points = new IntPoint[20000];
  GridLineTraversalLine line;
  line.points = points;

  IntPoint start = IntPoint(x0, y0);
  IntPoint end = IntPoint(x1, y1);

  if (supercover)
    GridLineTraversal::gridLine(start, end, &line);
  else
    GridLineTraversal::gridLineOld(start, end, &line);

  return line.num_points;
}

BOOST_AUTO_TEST_CASE(LineBresenhamTest) {
  // Check Bresenham algorithm for line discretization.
  bool supercover = false;

  BOOST_CHECK(bresenham_num_points(0, 0, 0, 0, supercover) == 1);       // Line contained in single cell
  BOOST_CHECK(bresenham_num_points(0, 0, 10, 10, supercover) == 11);    // First Quadrant 45º Line
  BOOST_CHECK(bresenham_num_points(-8, 12, -2, 5, supercover) == 8);    // Second Quadrant Line
  BOOST_CHECK(bresenham_num_points(-15, -3, -6, 0, supercover) == 10);  // Third Quadrant Line
  BOOST_CHECK(bresenham_num_points(-3, -4, 0, 1, supercover) == 6);     // Fourth Quadrant Line
  BOOST_CHECK(bresenham_num_points(-2, 3, 9, 11, supercover) ==
              12);  // Tricky for Bresenham (misses about 40% of the cells crossed by the line!)
  BOOST_CHECK(bresenham_num_points(4, 9, 14, 14, supercover) == 11);                 // Slope of 1/2
  BOOST_CHECK(bresenham_num_points(80, -25, 133, 41, supercover) == 67);             // Slope of 2
  BOOST_CHECK(bresenham_num_points(6, 3, 14, 14, supercover) == 12);                 // Same as Above +90º
  BOOST_CHECK(bresenham_num_points(-100, 1, 200, 1, supercover) == 301);             // Long Line
  BOOST_CHECK(bresenham_num_points(-2024, 3029, 9085, 11120, supercover) == 11110);  // Even longer Line
}

BOOST_AUTO_TEST_CASE(LineSuperCoverTest) {
  // Check Line SuperCover algorithm for line discretization.
  bool supercover = true;

  BOOST_CHECK(bresenham_num_points(0, 0, 0, 0, supercover) == 1);       // Line contained in single cell
  BOOST_CHECK(bresenham_num_points(0, 0, 10, 10, supercover) == 11);    // First Quadrant 45º Line
  BOOST_CHECK(bresenham_num_points(-8, 12, -2, 5, supercover) == 14);   // Second Quadrant Line
  BOOST_CHECK(bresenham_num_points(-15, -3, -6, 0, supercover) == 10);  // Third Quadrant Line
  BOOST_CHECK(bresenham_num_points(-3, -4, 0, 1, supercover) == 8);     // Fourth Quadrant Line
  BOOST_CHECK(bresenham_num_points(-2, 3, 9, 11, supercover) ==
              20);  // Tricky for Bresenham (misses about 40% of the cells crossed by the line!)
  BOOST_CHECK(bresenham_num_points(4, 9, 14, 14, supercover) == 16);                 // Slope of 1/2
  BOOST_CHECK(bresenham_num_points(80, -25, 133, 41, supercover) == 120);            // Slope of 2
  BOOST_CHECK(bresenham_num_points(6, 3, 14, 14, supercover) == 20);                 // Same as Above +90º
  BOOST_CHECK(bresenham_num_points(-100, 1, 200, 1, supercover) == 301);             // Long Line
  BOOST_CHECK(bresenham_num_points(-2024, 3029, 9085, 11120, supercover) == 19198);  // Even longer Line
}
