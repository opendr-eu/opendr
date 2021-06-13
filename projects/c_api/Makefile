#
# Copyright 2020-2021 OpenDR project
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


CPP = g++
CC = gcc
AR = ar
WGET = wget -r -q
MV = mv

MKDIR_P = mkdir -p
BUILD_DIR = build
DATA_DIR = data
LIB_DIR = lib

CFLAGS  = -g
INC = -I/usr/local/include/onnxruntime/core/session/ `pkg-config --cflags opencv4`
LD =  -L/usr/lib/ -L/usr/local/lib/ -lstdc++ -lm -lonnxruntime `pkg-config --libs opencv4` -lboost_filesystem
OPENDR_INC = -I./include
OPENDR_LD = -L./lib -lopendr

all: install_dependencies download libopendr demos runtests

libopendr:
	@+echo "Building C API..."
	@+$(MKDIR_P) $(BUILD_DIR)
	$(CPP) $(CFLAGS) -c  src/opendr_utils.cpp -o $(BUILD_DIR)/opendr_utils.o $(INC) $(OPENDR_INC)
	$(CPP) $(CFLAGS) -c  src/face_recognition.cpp -o $(BUILD_DIR)/opendr_face_recognition.o $(INC) $(OPENDR_INC)
	@$(MKDIR_P) $(LIB_DIR)
	$(AR) crf lib/libopendr.a $(BUILD_DIR)/opendr_utils.o $(BUILD_DIR)/opendr_face_recognition.o

demos:
	@+echo "Building C API demos..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/face_recognition_demo samples/face_recognition/face_recognition_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

download:
	@+echo "Downloading resources for C API..."
	@+$(MKDIR_P) $(DATA_DIR)
	@+$(WGET) ftp://opendrdata.csd.auth.gr/perception/face_recognition/test_data/*
	@+$(MV) opendrdata.csd.auth.gr/perception/face_recognition/test_data/images  $(DATA_DIR)/database
	@+$(WGET) ftp://opendrdata.csd.auth.gr/perception/face_recognition/optimized_model/*
	@+$(MV) opendrdata.csd.auth.gr/perception/face_recognition/optimized_model  $(DATA_DIR)/optimized_model
	@+$(RM) -r opendrdata.csd.auth.gr

runtests:
	@+echo "Building and running C API tests..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_opendr_utils tests/test_opendr_utils.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_face_recognition tests/test_face_recognition.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

	@+echo "Running test_opendr_utils..."
	@+$(BUILD_DIR)/test_opendr_utils 2>/dev/null

	@+echo "Running test_face_recognition..."
	@+$(BUILD_DIR)/test_face_recognition 2>/dev/null

	@+echo "Tests completed successfully!"

clean:
	@+echo "Cleaning C API binaries and temporary files..."
	@+$(RM) $(BUILD_DIR)/*
	@+$(RM) $(LIB_DIR)/*
	@+$(RM) -r $(DATA_DIR)/*
	@+echo "Done!"

install_dependencies:
	./install_deps.sh
