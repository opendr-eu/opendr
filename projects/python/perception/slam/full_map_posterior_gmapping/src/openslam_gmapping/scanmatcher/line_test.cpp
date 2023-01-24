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

#include <chrono>
#include <iostream>
#include "gmapping/scanmatcher/gridlinetraversal.h"

using namespace std;
using namespace GMapping;

int main(int argc, const char *const *argv) {
  IntPoint *old_line_points = new IntPoint[20000];
  IntPoint *line_points = new IntPoint[20000];

  GridLineTraversalLine old_line;
  old_line.points = old_line_points;

  GridLineTraversalLine line;
  line.points = line_points;

  int i, j;
  int test_times = 100000;

  int tests[][4] = {
    //     x1 ,   y1 ,   x2 ,   y2
    {0, 0, 0, 0},                // Line contained in single cell
    {0, 0, 10, 10},              // First Quadrant 45º Line
    {-8, 12, -2, 5},             // Second Quadrant Line
    {-15, -3, -6, 0},            // Third Quadrant Line
    {-3, -4, 0, 1},              // Fourth Quadrant Line
    {-2, 3, 9, 11},              // Tricky for Bresenham (misses about 40% of the cells crossed by the line!)
    {4, 9, 14, 14},              // Slope of 1/2
    {80, -25, 133, 41},          // Slope of 2
    {6, 3, 14, 14},              // Same as above +90º
    {-100, 1, 200, 1},           // Long Line
    {-2024, 3029, 9085, 11120},  // Even Longer Line
  };

  for (j = 0; j < sizeof(tests) / sizeof(tests[0]); j++) {
    long line_duration = 0, old_line_duration = 0;
    IntPoint start = IntPoint(tests[j][0], tests[j][1]);
    IntPoint end = IntPoint(tests[j][2], tests[j][3]);

    for (i = 0; i < test_times; i++) {
      auto t1_line = std::chrono::high_resolution_clock::now();
      GridLineTraversal::gridLine(start, end, &line);
      auto t2_line = std::chrono::high_resolution_clock::now();
      line_duration += std::chrono::duration_cast<std::chrono::microseconds>(t2_line - t1_line).count();

      auto t1_old_line = std::chrono::high_resolution_clock::now();
      GridLineTraversal::gridLineOld(start, end, &old_line);
      auto t2_old_line = std::chrono::high_resolution_clock::now();
      old_line_duration += std::chrono::duration_cast<std::chrono::microseconds>(t2_old_line - t1_old_line).count();
    }
    double avg_line_duration = (double)(line_duration) / test_times;
    double avg_old_line_duration = (double)(old_line_duration) / test_times;

    cout << endl;

    cout << "Test #" << j + 1 << " with points P1=(" << start.x << ", " << start.y;
    cout << ") and P2=(" << end.x << ", " << end.y << ")" << endl;

    cout << "Durations: LineSuperCover: " << avg_line_duration << "us, Original Alg: " << avg_old_line_duration << "us."
         << endl;
    cout << "Num Points: LineSuperCover: " << line.num_points << ", Original Alg: " << old_line.num_points << endl;
  }
}
