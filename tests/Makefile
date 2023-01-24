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
		$(MKDIR_P) $(DATA_DIR)/activity_recognition/x3d; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/activity_recognition/weights/optimized_model/*; \
		$(MV) opendrdata.csd.auth.gr/perception/activity_recognition/weights/optimized_model  $(DATA_DIR)/activity_recognition/x3d/optimized_model; \
		$(MKDIR_P) $(DATA_DIR); \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/face_recognition/test_data/*; \
		$(MV) opendrdata.csd.auth.gr/perception/face_recognition/test_data/images  $(DATA_DIR)/database; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/face_recognition/optimized_model/*; \
		$(MV) opendrdata.csd.auth.gr/perception/face_recognition/optimized_model  $(DATA_DIR)/optimized_model; \
		$(MKDIR_P) $(DATA_DIR)/lightweight_open_pose; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/pose_estimation/lightweight_open_pose/optimized_model/*; \
		$(MV) opendrdata.csd.auth.gr/perception/pose_estimation/lightweight_open_pose/optimized_model  $(DATA_DIR)/lightweight_open_pose/optimized_model; \
		$(MKDIR_P) $(DATA_DIR)/object_detection_2d/detr; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/object_detection_2d/detr/optimized_model/*; \
		$(MV) opendrdata.csd.auth.gr/perception/object_detection_2d/detr/optimized_model  $(DATA_DIR)/object_detection_2d/detr/optimized_model; \
		$(MKDIR_P) $(DATA_DIR)/object_detection_2d/nanodet; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/object_detection_2d/nanodet/images/*; \
		$(MV) opendrdata.csd.auth.gr/perception/object_detection_2d/nanodet/images  $(DATA_DIR)/object_detection_2d/nanodet/database; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/object_detection_2d/nanodet/optimized_model/*; \
		$(MV) opendrdata.csd.auth.gr/perception/object_detection_2d/nanodet/optimized_model  $(DATA_DIR)/object_detection_2d/nanodet/optimized_model; \
		$(MKDIR_P) $(DATA_DIR)/object_tracking_2d/deep_sort; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/object_tracking_2d/deep_sort/optimized_model/*; \
		$(MV) opendrdata.csd.auth.gr/perception/object_tracking_2d/deep_sort/optimized_model  $(DATA_DIR)/object_tracking_2d/deep_sort/optimized_model; \
		$(MKDIR_P) $(DATA_DIR)/skeleton_based_action_recognition/progressive_spatiotemporal_gcn; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/skeleton_based_action_recognition/pretrained_models/optimized_data/*; \
		$(MV) opendrdata.csd.auth.gr/perception/skeleton_based_action_recognition/pretrained_models/optimized_data  $(DATA_DIR)/skeleton_based_action_recognition/progressive_spatiotemporal_gcn/optimized_model; \
		$(RM) -r opendrdata.csd.auth.gr; \
	fi;

$(BUILD_DIR)/test_opendr_utils:
	@+echo "Building C API test..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_opendr_utils sources/c_api/test_opendr_utils.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/test_activity_recognition_x3d:
	@+echo "Building activity recognition x3d demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_activity_recognition_x3d sources/c_api/test_activity_recognition_x3d.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/test_face_recognition:
	@+echo "Building face recognition test..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_face_recognition sources/c_api/test_face_recognition.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/test_lightweight_open_pose:
	@+echo "Building lightweight open pose demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_lightweight_open_pose sources/c_api/test_lightweight_open_pose.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/test_detr:
	@+echo "Building detr object detection demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_detr sources/c_api/test_detr.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/test_nanodet:
	@+echo "Building Nanodet object detection test..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_nanodet sources/c_api/test_nanodet.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/test_deep_sort:
	@+echo "Building object tracking 2d deep sort demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_deep_sort sources/c_api/test_deep_sort.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/test_pst:
	@+echo "Building skeleton based action recognition progressive spatiotemporal demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/test_pst sources/c_api/test_pst_gcn.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)


FMP_INC = -I$(OPENDR_HOME)/projects/python/perception/slam/full_map_posterior_gmapping/src/openslam_gmapping/include
$(BUILD_DIR)/test_fmp_gmapping:
	@+echo "Building Full-Map-Posterior GMapping test..."
	$(CPP)  $(CFLAGS) -o $(BUILD_DIR)/test_fmp_gmapping sources/c_api/test_fmp_gmapping.cpp -lboost_unit_test_framework $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD) $(FMP_INC)


tests: utils x3d face_recognition open_pose detr nanodet deep_sort skeleton_based_action_recognition_pst fmp_gmapping

utils: $(BUILD_DIR)/test_opendr_utils
x3d: $(BUILD_DIR)/test_activity_recognition_x3d
face_recognition: $(BUILD_DIR)/test_face_recognition
open_pose: $(BUILD_DIR)/test_lightweight_open_pose
detr: $(BUILD_DIR)/test_detr
nanodet: $(BUILD_DIR)/test_nanodet
deep_sort: $(BUILD_DIR)/test_deep_sort
skeleton_based_action_recognition_pst: $(BUILD_DIR)/test_pst
fmp_gmapping: $(BUILD_DIR)/test_fmp_gmapping

runtests: download tests
	@+$(LD_RUN) $(BUILD_DIR)/test_opendr_utils 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_activity_recognition_x3d 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_face_recognition 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_lightweight_open_pose 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_detr 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_nanodet 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_deep_sort 2>/dev/null
	@+$(LD_RUN) $(BUILD_DIR)/test_pst 2>/dev/null -T 1800
	@+$(LD_RUN) $(BUILD_DIR)/test_fmp_gmapping

clean:
	@+echo "Cleaning C tests binaries and temporary files..."
	@+$(RM) $(BUILD_DIR)/test_opendr_utils
	@+$(RM) $(BUILD_DIR)/test_activity_recognition_x3d
	@+$(RM) $(BUILD_DIR)/test_face_recognition
	@+$(RM) $(BUILD_DIR)/test_lightweight_open_pose
	@+$(RM) $(BUILD_DIR)/test_detr
	@+$(RM) $(BUILD_DIR)/test_nanodet
	@+$(RM) $(BUILD_DIR)/test_deep_sort
	@+$(RM) $(BUILD_DIR)/test_pst
	@+$(RM) $(BUILD_DIR)/test_fmp_gmapping
	@+$(RM) -rf $(DATA_DIR)
	@+echo "Done!"
