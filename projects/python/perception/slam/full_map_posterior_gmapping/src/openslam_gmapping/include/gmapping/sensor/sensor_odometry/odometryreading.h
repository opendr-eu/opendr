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

#ifndef ODOMETRYREADING_H
#define ODOMETRYREADING_H

#include <gmapping/sensor/sensor_base/sensorreading.h>
#include <gmapping/utils/point.h>
#include <string.h>
#include "gmapping/sensor/sensor_odometry/odometrysensor.h"

namespace GMapping {

  class OdometryReading : public SensorReading {
  public:
    OdometryReading(const OdometrySensor *odo, double time = 0);

    inline const OrientedPoint &getPose() const { return m_pose; }

    inline const OrientedPoint &getSpeed() const { return m_speed; }

    inline const OrientedPoint &getAcceleration() const { return m_acceleration; }

    inline void setPose(const OrientedPoint &pose) { m_pose = pose; }

    inline void setSpeed(const OrientedPoint &speed) { m_speed = speed; }

    inline void setAcceleration(const OrientedPoint &acceleration) { m_acceleration = acceleration; }

  protected:
    OrientedPoint m_pose;
    OrientedPoint m_speed;
    OrientedPoint m_acceleration;
  };

};  // namespace GMapping
#endif
