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

#include "gmapping/sensor/sensor_range/rangesensor.h"

namespace GMapping {

  RangeSensor::RangeSensor(std::string name) : Sensor(name) {}

  RangeSensor::RangeSensor(std::string name, unsigned int beams_num, double res, const OrientedPoint &position, double span,
                           double maxrange) :
    Sensor(name),
    m_pose(position),
    m_beams(beams_num) {
    double angle = -res * (int)(beams_num / 2);
    for (unsigned int i = 0; i < beams_num; i++, angle += res) {
      RangeSensor::Beam &beam(m_beams[i]);
      beam.span = span;
      beam.pose.x = 0;
      beam.pose.y = 0;
      beam.pose.theta = angle;
      beam.maxRange = maxrange;
    }
    newFormat = 0;
    updateBeamsLookup();
  }

  void RangeSensor::updateBeamsLookup() {
    for (unsigned int i = 0; i < m_beams.size(); i++) {
      RangeSensor::Beam &beam(m_beams[i]);
      beam.s = sin(m_beams[i].pose.theta);
      beam.c = cos(m_beams[i].pose.theta);
    }
  }

};  // namespace GMapping
