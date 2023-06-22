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

#ifndef RANGESENSOR_H
#define RANGESENSOR_H

#include <gmapping/sensor/sensor_base/sensor.h>
#include <gmapping/utils/point.h>
#include <vector>

namespace GMapping {

  class RangeSensor : public Sensor {
    friend class Configuration;

    friend class CarmenConfiguration;

    friend class CarmenWrapper;

  public:
    struct Beam {
      OrientedPoint pose;  // pose relative to the center of the sensor
      double span;         // spam=0 indicates a line-like beam
      double maxRange;     // maximum range of the sensor
      double s, c;         // sinus and cosinus of the beam (optimization);
    };

    RangeSensor(std::string name);

    RangeSensor(std::string name, unsigned int beams, double res, const OrientedPoint &position = OrientedPoint(0, 0, 0),
                double span = 0, double maxrange = 89.0);

    inline const std::vector<Beam> &beams() const { return m_beams; }

    inline std::vector<Beam> &beams() { return m_beams; }

    inline OrientedPoint getPose() const { return m_pose; }

    void updateBeamsLookup();

    bool newFormat;

  protected:
    OrientedPoint m_pose;
    std::vector<Beam> m_beams;
  };

};  // namespace GMapping

#endif
