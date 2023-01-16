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

#ifndef FSRMOVEMENT_H
#define FSRMOVEMENT_H

#include "gmapping/utils/point.h"

namespace GMapping {

  /** fsr-movement (forward, sideward, rotate) **/
  class FSRMovement {
  public:
    FSRMovement(double f = 0.0, double s = 0.0, double r = 0.0);

    FSRMovement(const FSRMovement &src);

    FSRMovement(const OrientedPoint &pt1, const OrientedPoint &pt2);

    FSRMovement(const FSRMovement &move1, const FSRMovement &move2);

    void normalize();

    void invert();

    void compose(const FSRMovement &move2);

    OrientedPoint move(const OrientedPoint &pt) const;

    /* static members */

    static OrientedPoint movePoint(const OrientedPoint &pt, const FSRMovement &move1);

    static FSRMovement composeMoves(const FSRMovement &move1, const FSRMovement &move2);

    static FSRMovement moveBetweenPoints(const OrientedPoint &pt1, const OrientedPoint &pt2);

    static FSRMovement invertMove(const FSRMovement &move1);

    static OrientedPoint frameTransformation(const OrientedPoint &reference_pt_frame1, const OrientedPoint &reference_pt_frame2,
                                             const OrientedPoint &pt_frame1);

  public:
    double f;
    double s;
    double r;
  };
}  // namespace GMapping
#endif
