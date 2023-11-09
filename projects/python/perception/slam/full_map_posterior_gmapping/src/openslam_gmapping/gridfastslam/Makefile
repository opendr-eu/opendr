# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OBJS= gridslamprocessor_tree.o motionmodel.o gridslamprocessor.o gfsreader.o
APPS= gfs2log gfs2rec gfs2neff #gfs2stat

#LDFLAGS+= -lutils -lsensor_range -llog -lscanmatcher -lsensor_base -lsensor_odometry $(GSL_LIB)
LDFLAGS+=  -lscanmatcher -llog -lsensor_range -lsensor_odometry -lsensor_base -lutils
#CPPFLAGS+=-I../sensor $(GSL_INCLUDE)
CPPFLAGS+=-I../sensor

-include ../global.mk
-include ../build_tools/Makefile.generic-shared-object
