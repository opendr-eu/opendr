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
DATA_DIR = data
LIB_DIR = $(OPENDR_HOME)/lib

CFLAGS  = -fPIC
INC = -I/usr/local/include/onnxruntime/core/session/ `pkg-config --cflags opencv4`
LD =  -L/usr/lib/ -L/usr/local/lib/ -lstdc++ -lm -lonnxruntime `pkg-config --libs opencv4` -lboost_filesystem  `pkg-config --cflags --libs check`
OPENDR_INC = -I$(OPENDR_HOME)/include
OPENDR_LD = -L$(OPENDR_HOME)/lib -lopendr
LD_RUN = LD_LIBRARY_PATH=$(OPENDR_HOME)/lib:$(LD_LIBRARY_PATH)

all: tests

download:
	@+if [ -a $(DATA_DIR) ] ; \
	then \
		echo Resources already downloaded. Run rm -rf $(DATA_DIR) to remove existing resources.; \
	else \
		echo Downloading resources for C API...; \
		$(RM) -r opendrdata.csd.auth.gr; \
		$(MKDIR_P) $(DATA_DIR); \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/face_recognition/test_data/*; \
		$(MV) opendrdata.csd.auth.gr/perception/face_recognition/test_data/images  $(DATA_DIR)/database; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/face_recognition/optimized_model/*; \
		$(MV) opendrdata.csd.auth.gr/perception/face_recognition/optimized_model  $(DATA_DIR)/optimized_model; \
		$(RM) -r opendrdata.csd.auth.gr; \
	fi;

$(BUILD_DIR)/test_opendr_utils:
	@+echo "Building C API test..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_opendr_utils sources/c_api/test_opendr_utils.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/test_face_recognition:
	@+echo "Building face recognition test..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_face_recognition sources/c_api/test_face_recognition.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

FMP_INC = -I$(OPENDR_HOME)/projects/python/perception/slam/full_map_posterior_gmapping/src/openslam_gmapping/include
$(BUILD_DIR)/test_fmp_gmapping:
	@+echo "Building Full-Map-Posterior GMapping test..."
	$(CPP)  $(CFLAGS) -o $(BUILD_DIR)/test_fmp_gmapping sources/c_api/test_fmp_gmapping.cpp -lboost_unit_test_framework $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD) $(FMP_INC)

tests: $(BUILD_DIR)/test_opendr_utils $(BUILD_DIR)/test_face_recognition $(BUILD_DIR)/test_fmp_gmapping

runtests: download tests
	@+$(LD_RUN) $(BUILD_DIR)/test_opendr_utils 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_face_recognition 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_fmp_gmapping

clean:
	@+echo "Cleaning C tests binaries and temporary files..."
	@+$(RM) $(BUILD_DIR)/test_opendr_utils
	@+$(RM) $(BUILD_DIR)/test_face_recognition
	@+$(RM) $(BUILD_DIR)/test_fmp_gmapping
	@+$(RM) -rf $(DATA_DIR)
	@+echo "Done!"
