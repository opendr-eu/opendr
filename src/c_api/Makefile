#
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

SHELL := /bin/bash
CPP = g++
CC = gcc
AR = ar
WGET = wget -r -q
MV = mv

MKDIR_P = mkdir -p
BUILD_DIR = $(OPENDR_HOME)/build
LIB_DIR = $(OPENDR_HOME)/lib

CFLAGS  = -fPIC
INC = -I/usr/local/include/onnxruntime/ `pkg-config --cflags opencv4`
LD =  -L/usr/lib/ -L/usr/local/lib/ -lstdc++ -lm -lonnxruntime `pkg-config --libs opencv4` -lboost_filesystem
OPENDR_INC = -I$(OPENDR_HOME)/include
OPENDR_LD = -L$(OPENDR_HOME)//lib -lopendr

all: libopendr

libopendr: $(OPENDR_HOME)/lib/libopendr.so

$(OPENDR_HOME)/lib/libopendr.so:
	@+echo "Building C API..."
	@+$(MKDIR_P) $(BUILD_DIR)
	$(CPP) $(CFLAGS) -c opendr_utils.cpp -o $(BUILD_DIR)/opendr_utils.o $(INC) $(OPENDR_INC)
	$(CPP) $(CFLAGS) -c face_recognition.cpp -o $(BUILD_DIR)/opendr_face_recognition.o $(INC) $(OPENDR_INC)
	@$(MKDIR_P) $(LIB_DIR)
	$(CPP) $(CFLAGS)  -o $(OPENDR_HOME)/lib/libopendr.so $(BUILD_DIR)/opendr_utils.o $(BUILD_DIR)/opendr_face_recognition.o $(LD)  --shared

clean:
	@+echo "Cleaning C API binaries and temporary files..."
	@+$(RM) $(BUILD_DIR)/*
	@+$(RM) $(LIB_DIR)/*.so
	@+echo "Done!"
