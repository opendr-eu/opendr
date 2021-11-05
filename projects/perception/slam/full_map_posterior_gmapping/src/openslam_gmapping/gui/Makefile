
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

-include ../global.mk

OBJS= gsp_thread.o qparticleviewer.o qgraphpainter.o qmappainter.o 

APPS= gfs_nogui gfs_simplegui gfs2img
LDFLAGS+=  $(QT_LIB) $(KDE_LIB) -lgridfastslam -lscanmatcher -llog -lsensor_range -lsensor_odometry -lsensor_base -lconfigfile -lutils -lpthread

ifeq ($(CARMENSUPPORT),1)
LDFLAGS+= -lcarmenwrapper 
endif

CPPFLAGS+= -I../sensor $(QT_INCLUDE) $(KDE_INCLUDE) -I$(CARMEN_HOME)/include


-include ../build_tools/Makefile.generic-shared-object



