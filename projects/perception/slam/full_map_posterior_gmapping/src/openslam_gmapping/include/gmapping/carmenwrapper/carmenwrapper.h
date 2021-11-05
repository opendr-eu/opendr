/*****************************************************************
 *
 * This file is part of the GMAPPING project
 *
 * GMAPPING Copyright (c) 2004 Giorgio Grisetti,
 * Cyrill Stachniss, and Wolfram Burgard
 *
 * This software is licensed under the "Creative Commons
 * License (Attribution-NonCommercial-ShareAlike 2.0)"
 * and is copyrighted by Giorgio Grisetti, Cyrill Stachniss,
 * and Wolfram Burgard.
 *
 * Further information on this license can be found at:
 * http://creativecommons.org/licenses/by-nc-sa/2.0/
 *
 * GMAPPING is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.
 *
 *****************************************************************/

/*
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

#ifndef CARMENWRAPPER_H
#define CARMENWRAPPER_H

#include <carmen/carmen.h>
#include <carmen/global.h>
#include <gmapping/log/carmenconfiguration.h>
#include <gmapping/log/sensorlog.h>
#include <gmapping/log/sensorstream.h>
#include <gmapping/sensor/sensor_base/sensor.h>
#include <gmapping/sensor/sensor_range/rangereading.h>
#include <gmapping/sensor/sensor_range/rangesensor.h>
#include <pthread.h>
#include <semaphore.h>
#include <deque>
#include <iostream>

namespace GMapping {

  class CarmenWrapper {
  public:
    static void initializeIPC(const char *name);

    static bool start(const char *name);

    static bool isRunning();

    static void lock();

    static void unlock();

    static int registerLocalizationMessages();

    static int queueLength();

    static OrientedPoint getTruePos();

    static bool getReading(RangeReading &reading);

    static void addReading(RangeReading &reading);

    static const SensorMap &sensorMap();

    static bool sensorMapComputed();

    static bool isStopped();

    // conversion function
    static carmen_robot_laser_message reading2carmen(const RangeReading &reading);

    static RangeReading carmen2reading(const carmen_robot_laser_message &msg);

    static carmen_point_t point2carmen(const OrientedPoint &p);

    static OrientedPoint carmen2point(const carmen_point_t &p);

    // carmen interaction
    static void robot_frontlaser_handler(carmen_robot_laser_message *frontlaser);

    static void robot_rearlaser_handler(carmen_robot_laser_message *frontlaser);

    static void simulator_truepos_handler(carmen_simulator_truepos_message *truepos);

    // babsi:
    static void navigator_go_handler(MSG_INSTANCE msgRef, BYTE_ARRAY callData, void *);

    static void navigator_stop_handler(MSG_INSTANCE msgRef, BYTE_ARRAY callData, void *);

    // babsi:
    static void publish_globalpos(carmen_localize_summary_p summary);

    static void publish_particles(carmen_localize_particle_filter_p filter, carmen_localize_summary_p summary);

    static void shutdown_module(int sig);

  private:
    static std::deque<RangeReading> m_rangeDeque;
    static sem_t m_dequeSem;
    static pthread_mutex_t m_mutex, m_lock;
    static pthread_t m_readingThread;

    static void *m_reading_function(void *);

    static bool m_threadRunning;
    static SensorMap m_sensorMap;
    static RangeSensor *m_frontLaser, *m_rearLaser;
    static OrientedPoint m_truepos;
    static bool stopped;
  };

}  // namespace GMapping

#endif
/*
int main (int argc, char** argv) {

        CarmenWrapper::init_carmen(argc, argv);
        while (true) {
                IPC_listenWait(100);
        }
        return 1;
}
*/
