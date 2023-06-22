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

#ifndef GRIDLINETRAVERSAL_H
#define GRIDLINETRAVERSAL_H

#include <gmapping/utils/point.h>
#include <cstdlib>

namespace GMapping {

  typedef struct {
    int num_points;
    IntPoint *points;
  } GridLineTraversalLine;

  struct GridLineTraversal {
    inline static void gridLine(IntPoint start, IntPoint end, GridLineTraversalLine *line);

    inline static void gridLineOld(IntPoint start, IntPoint end, GridLineTraversalLine *line);

    inline static void gridLineCore(IntPoint start, IntPoint end, GridLineTraversalLine *line);
  };

  void GridLineTraversal::gridLineCore(IntPoint start, IntPoint end, GridLineTraversalLine *line) {
    int dx, dy, incr1, incr2, d, x, y, xend, yend, xdirflag, ydirflag;
    int cnt = 0;

    dx = abs(end.x - start.x);
    dy = abs(end.y - start.y);

    if (dy <= dx) {
      d = 2 * dy - dx;
      incr1 = 2 * dy;
      incr2 = 2 * (dy - dx);
      if (start.x > end.x) {
        x = end.x;
        y = end.y;
        ydirflag = (-1);
        xend = start.x;
      } else {
        x = start.x;
        y = start.y;
        ydirflag = 1;
        xend = end.x;
      }
      line->points[cnt].x = x;
      line->points[cnt].y = y;
      cnt++;
      if (((end.y - start.y) * ydirflag) > 0) {
        while (x < xend) {
          x++;
          if (d < 0) {
            d += incr1;
          } else {
            y++;
            d += incr2;
          }
          line->points[cnt].x = x;
          line->points[cnt].y = y;
          cnt++;
        }
      } else {
        while (x < xend) {
          x++;
          if (d < 0) {
            d += incr1;
          } else {
            y--;
            d += incr2;
          }
          line->points[cnt].x = x;
          line->points[cnt].y = y;
          cnt++;
        }
      }
    } else {
      d = 2 * dx - dy;
      incr1 = 2 * dx;
      incr2 = 2 * (dx - dy);
      if (start.y > end.y) {
        y = end.y;
        x = end.x;
        yend = start.y;
        xdirflag = (-1);
      } else {
        y = start.y;
        x = start.x;
        yend = end.y;
        xdirflag = 1;
      }
      line->points[cnt].x = x;
      line->points[cnt].y = y;
      cnt++;
      if (((end.x - start.x) * xdirflag) > 0) {
        while (y < yend) {
          y++;
          if (d < 0) {
            d += incr1;
          } else {
            x++;
            d += incr2;
          }
          line->points[cnt].x = x;
          line->points[cnt].y = y;
          cnt++;
        }
      } else {
        while (y < yend) {
          y++;
          if (d < 0) {
            d += incr1;
          } else {
            x--;
            d += incr2;
          }
          line->points[cnt].x = x;
          line->points[cnt].y = y;
          cnt++;
        }
      }
    }
    line->num_points = cnt;
  }

  void GridLineTraversal::gridLineOld(IntPoint start, IntPoint end, GridLineTraversalLine *line) {
    int i, j;
    int half;
    IntPoint v;
    gridLineCore(start, end, line);
    if (start.x != line->points[0].x || start.y != line->points[0].y) {
      half = line->num_points / 2;
      for (i = 0, j = line->num_points - 1; i < half; i++, j--) {
        v = line->points[i];
        line->points[i] = line->points[j];
        line->points[j] = v;
      }
    }
  }

  void GridLineTraversal::gridLine(IntPoint start, IntPoint end, GridLineTraversalLine *line) {
    /*
     * Bresenham-based supercover line algorithm
     * Unlike pure Bresenham's, this algorithm covers all grid cells for a given line.
     * It is therefore a bit less time efficient, and requires more line points in the end,
     * but all those line points are traversed by the line, whereas Bresenham's only returns one per x step.
     *
     * Based on http://eugen.dedu.free.fr/projects/bresenham/ by Eugen Dedu
     */
    int i, p = 1;                                    // Iteration and point Counter.
    int x_step, y_step;                              // Step direction in x and y axes.
    int error, prev_error;                           // Accumulated error and previous value of error.
    int x = start.x, y = start.y;                    // Current Line points.
    int dx = end.x - start.x, dy = end.y - start.y;  // Line length in x and y components.
    int ddx, ddy;                                    // Double values of dx and dy for full precision.

    // Initialization
    // Add first point.
    line->points[0].x = x;
    line->points[0].y = y;

    x_step = dx < 0 ? -1 : 1;
    dx = dx < 0 ? -dx : dx;
    y_step = dy < 0 ? -1 : 1;
    dy = dy < 0 ? -dy : dy;

    ddx = 2 * dx;
    ddy = 2 * dy;

    // Differential Analysis
    // Due to symmetry, we only need to check the Top-Right Quadrant.
    if (ddx >= ddy) {  // Angle in [0º, 45º].
      prev_error = error = dx;
      for (i = 0; i < dx; i++) {
        x += x_step;
        error += ddy;

        if (error > ddx) {
          y += y_step;
          error -= ddx;

          if (error + prev_error < ddx) {  // Also include the cell below.
            line->points[p].x = x;
            line->points[p].y = y - y_step;
            p++;
          } else if (error + prev_error > ddx) {  // Also include the cell to the left.
            line->points[p].x = x - x_step;
            line->points[p].y = y;
            p++;
          }
          // Ignoring the case where the Line hits exactly a grid corner (ddx == error + prev_error).
          /*
          else { // Also include the cell below and the cell to the left.
              line->points[p].x = x;
              line->points[p].y = y - y_step;
              line->points[p].x = x - x_step;
              line->points[p].y = y;
              p += 2;
          }
          */
        }
        // Include current cell
        line->points[p].x = x;
        line->points[p].y = y;
        p++;
        prev_error = error;
      }
    } else {  // Angle in (45º, 90º)
      prev_error = error = dy;

      for (i = 0; i < dy; i++) {
        y += y_step;
        error += ddx;

        if (error > ddy) {
          x += x_step;
          error -= ddy;

          if (error + prev_error < ddy) {  // Also include the cell to the left
            line->points[p].x = x - x_step;
            line->points[p].y = y;
            p++;
          } else if (error + prev_error > ddy) {  // Also include the cell below
            line->points[p].x = x;
            line->points[p].y = y - y_step;
            p++;
          }
          // Ignoring the case where the Line hits exactly a grid corner (ddy == error + prev_error).
          /*
          else { // Also include the cell to the left and the cell below.
              line->points[p].x = x - x_step;
              line->points[p].y = y;
              line->points[p].x = x;
              line->points[p].y = y - y_step;
              p += 2;
          }
          */
        }
        line->points[p].x = x;
        line->points[p].y = y;
        p++;
        prev_error = error;
      }
    }

    line->num_points = p;
    // The end point has to be the equal to the last point of the algorithm.
    assert(line->points[line->num_points - 1] == end);
  }

};  // namespace GMapping

#endif
