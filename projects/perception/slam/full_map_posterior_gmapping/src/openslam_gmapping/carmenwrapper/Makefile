
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

OBJS= carmenwrapper.o
APPS= 

LIBS+= -L$(CARMEN_HOME)/lib -lnavigator_interface -lsimulator_interface -lrobot_interface -llaser_interface -lparam_interface -lglobal -lipc -lm -lutils -lsensor_range -llog -lscanmatcher -lpthread -lz
CPPFLAGS+=-I../sensor -I$(CARMEN_HOME)/include

-include ../global.mk

ifeq ($(CARMENSUPPORT), 0)
OBJS=
	-include ../build_tools/Makefile.app
else
	-include ../build_tools/Makefile.generic-shared-object
endif
