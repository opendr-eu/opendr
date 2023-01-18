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

#ifndef CARMENCONFIGURATION_H
#define CARMENCONFIGURATION_H

#include <gmapping/sensor/sensor_base/sensor.h>
#include <istream>
#include <map>
#include <string>
#include <vector>
#include "gmapping/log/configuration.h"

namespace GMapping {

  class CarmenConfiguration : public Configuration, public std::map<std::string, std::vector<std::string>> {
  public:
    virtual std::istream &load(std::istream &is);

    virtual SensorMap computeSensorMap() const;
  };

};  // namespace GMapping

#endif
