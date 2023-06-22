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

#ifndef SENSORSTREAM_H
#define SENSORSTREAM_H

#include <istream>
#include "gmapping/log/sensorlog.h"

namespace GMapping {
  class SensorStream {
  public:
    SensorStream(const SensorMap &sensorMap);

    virtual ~SensorStream();

    virtual operator bool() const = 0;

    virtual bool rewind() = 0;

    virtual SensorStream &operator>>(const SensorReading *&) = 0;

    inline const SensorMap &getSensorMap() const { return m_sensorMap; }

  protected:
    const SensorMap &m_sensorMap;

    static SensorReading *parseReading(std::istream &is, const SensorMap &smap);

    static OdometryReading *parseOdometry(std::istream &is, const OdometrySensor *);

    static RangeReading *parseRange(std::istream &is, const RangeSensor *);
  };

  class InputSensorStream : public SensorStream {
  public:
    InputSensorStream(const SensorMap &sensorMap, std::istream &is);

    virtual operator bool() const;

    virtual bool rewind();

    virtual SensorStream &operator>>(const SensorReading *&);

    // virtual SensorStream& operator >>(SensorLog*& log);
  protected:
    std::istream &m_inputStream;
  };

  class LogSensorStream : public SensorStream {
  public:
    LogSensorStream(const SensorMap &sensorMap, const SensorLog *log);

    virtual operator bool() const;

    virtual bool rewind();

    virtual SensorStream &operator>>(const SensorReading *&);

  protected:
    const SensorLog *m_log;
    SensorLog::const_iterator m_cursor;
  };

};  // namespace GMapping
#endif
