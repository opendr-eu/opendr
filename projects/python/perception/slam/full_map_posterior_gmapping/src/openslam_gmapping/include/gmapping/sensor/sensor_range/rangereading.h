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

#ifndef RANGEREADING_H
#define RANGEREADING_H

#include <gmapping/sensor/sensor_base/sensorreading.h>
#include <vector>
#include "gmapping/sensor/sensor_range/rangesensor.h"

namespace GMapping {

  class RangeReading : public SensorReading, public std::vector<double> {
  public:
    RangeReading(const RangeSensor *rs, double time = 0);

    RangeReading(unsigned int n_beams, const double *d, const RangeSensor *rs, double time = 0);

    virtual ~RangeReading();

    inline const OrientedPoint &getPose() const { return m_pose; }

    inline void setPose(const OrientedPoint &pose) { m_pose = pose; }

    unsigned int rawView(double *v, double density = 0.) const;

    std::vector<Point> cartesianForm(double maxRange = 1e6) const;

    unsigned int activeBeams(double density = 0.) const;

  protected:
    OrientedPoint m_pose;
  };

};  // namespace GMapping

#endif
