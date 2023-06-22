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

#ifndef SENSORREADING_H
#define SENSORREADING_H

#include "gmapping/sensor/sensor_base/sensor.h"

namespace GMapping {

  class SensorReading {
  public:
    SensorReading(const Sensor *s = 0, double time = 0);

    inline double getTime() const { return m_time; }

    inline const Sensor *getSensor() const { return m_sensor; }

  protected:
    double m_time;
    const Sensor *m_sensor;
  };

};  // namespace GMapping
#endif
