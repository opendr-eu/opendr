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

#ifndef SENSORLOG_H
#define SENSORLOG_H

#include <gmapping/sensor/sensor_base/sensorreading.h>
#include <gmapping/sensor/sensor_odometry/odometryreading.h>
#include <gmapping/sensor/sensor_odometry/odometrysensor.h>
#include <gmapping/sensor/sensor_range/rangereading.h>
#include <gmapping/sensor/sensor_range/rangesensor.h>
#include <istream>
#include <list>
#include "gmapping/log/configuration.h"

namespace GMapping {

  class SensorLog : public std::list<SensorReading *> {
  public:
    SensorLog(const SensorMap &);

    ~SensorLog();

    std::istream &load(std::istream &is);

    OrientedPoint boundingBox(double &xmin, double &ymin, double &xmax, double &ymax) const;

  protected:
    const SensorMap &m_sensorMap;

    OdometryReading *parseOdometry(std::istream &is, const OdometrySensor *) const;

    RangeReading *parseRange(std::istream &is, const RangeSensor *) const;
  };

};  // namespace GMapping

#endif
