#ifndef ODOMETRYSENSOR_H
#define ODOMETRYSENSOR_H

#include <gmapping/sensor/sensor_base/sensor.h>
#include <string>

namespace GMapping {

  class OdometrySensor : public Sensor {
  public:
    OdometrySensor(const std::string &name, bool ideal = false);

    inline bool isIdeal() const { return m_ideal; }

  protected:
    bool m_ideal;
  };

};  // namespace GMapping

#endif
