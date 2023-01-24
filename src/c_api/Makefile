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
INC = -I/usr/local/include/onnxruntime/ -I/usr/local/include/rapidjson/ `pkg-config --cflags opencv4`
LD =  -L/usr/lib/ -L/usr/local/lib/ -lstdc++ -lm -lonnxruntime `pkg-config --libs opencv4` -lboost_filesystem
OPENDR_INC = -I$(OPENDR_HOME)/include
OPENDR_LD = -L$(OPENDR_HOME)//lib -lopendr

LIBTORCH_DIR = /usr/local/libtorch
TORCHSCRIPT_INC = -I$(LIBTORCH_DIR)/include -I$(LIBTORCH_DIR)/include/torch/csrc/api/include
TORCHSCRIPT_LD = -L$(LIBTORCH_DIR)/lib -L$(LIBTORCH_DIR)/share -ltorchvision -ltorch

all: libopendr

libopendr: $(OPENDR_HOME)/lib/libopendr.so

$(OPENDR_HOME)/lib/libopendr.so:
	@+echo "Building C API..."
	@+$(MKDIR_P) $(BUILD_DIR)
	$(CPP) $(CFLAGS) -c opendr_utils.cpp -o $(BUILD_DIR)/opendr_utils.o $(INC) $(OPENDR_INC)
	$(CPP) $(CFLAGS) -c activity_recognition_x3d.cpp -o $(BUILD_DIR)/opendr_x3d.o $(INC) $(OPENDR_INC)
	$(CPP) $(CFLAGS) -c face_recognition.cpp -o $(BUILD_DIR)/opendr_face_recognition.o $(INC) $(OPENDR_INC)
	$(CPP) $(CFLAGS) -c lightweight_open_pose.cpp -o $(BUILD_DIR)/opendr_open_pose.o $(INC) $(OPENDR_INC)
	$(CPP) $(CFLAGS) -c object_detection_2d_detr.cpp -o $(BUILD_DIR)/opendr_detr.o $(INC) $(OPENDR_INC)
	$(CPP) $(CFLAGS) -c object_detection_2d_nanodet_jit.cpp -o $(BUILD_DIR)/opendr_nanodet_jit.o $(INC) $(OPENDR_INC) $(TORCHSCRIPT_INC)
	$(CPP) $(CFLAGS) -c object_tracking_2d_deep_sort.cpp -o $(BUILD_DIR)/opendr_deep_sort.o $(INC) $(OPENDR_INC)
	$(CPP) $(CFLAGS) -c skeleton_based_action_recognition_pst.cpp -o $(BUILD_DIR)/opendr_skeleton_based_action_recognition_pst.o $(INC) $(OPENDR_INC)

	@$(MKDIR_P) $(LIB_DIR)
	$(CPP) $(CFLAGS)  -o $(OPENDR_HOME)/lib/libopendr.so $(BUILD_DIR)/opendr_utils.o $(BUILD_DIR)/opendr_x3d.o $(BUILD_DIR)/opendr_face_recognition.o $(BUILD_DIR)/opendr_open_pose.o $(BUILD_DIR)/opendr_detr.o $(BUILD_DIR)/opendr_nanodet_jit.o $(BUILD_DIR)/opendr_deep_sort.o $(BUILD_DIR)/opendr_skeleton_based_action_recognition_pst.o $(LD) $(TORCHSCRIPT_LD) --shared
clean:
	@+echo "Cleaning C API binaries and temporary files..."
	@+$(RM) $(BUILD_DIR)/*
	@+$(RM) $(LIB_DIR)/*.so
	@+echo "Done!"
