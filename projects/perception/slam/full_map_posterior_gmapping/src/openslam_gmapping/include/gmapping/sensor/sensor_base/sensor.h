/*
 * Copyright 2020-2022 OpenDR European Project
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

#ifndef SENSOR_H
#define SENSOR_H

#include <map>
#include <string>

namespace GMapping {

  class Sensor {
  public:
    Sensor(const std::string &name = "");

    virtual ~Sensor();

    inline std::string getName() const { return m_name; }

    inline void setName(const std::string &name) { m_name = name; }

  protected:
    std::string m_name;
  };

  typedef std::map<std::string, Sensor *> SensorMap;

};  // namespace GMapping

#endif
