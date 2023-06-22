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

#ifndef MOTIONMODEL_H
#define MOTIONMODEL_H

#include <gmapping/utils/macro_params.h>
#include <gmapping/utils/point.h>
#include <gmapping/utils/stat.h>

namespace GMapping {

  struct MotionModel {
    OrientedPoint drawFromMotion(const OrientedPoint &p, double linearMove, double angularMove) const;

    OrientedPoint drawFromMotion(const OrientedPoint &p, const OrientedPoint &pnew, const OrientedPoint &pold) const;

    Covariance3 gaussianApproximation(const OrientedPoint &pnew, const OrientedPoint &pold) const;

    double srr, str, srt, stt;
  };

};  // namespace GMapping

#endif
