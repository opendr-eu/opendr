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
BUILD_DIR = build
LIB_DIR = $(OPENDR_HOME)/lib
DATA_DIR = ./data

CFLAGS  = -fPIC
INC = -I/usr/local/include/onnxruntime/core/session/ `pkg-config --cflags opencv4`
LD =  -L/usr/lib/ -L/usr/local/lib/ -lstdc++ -lm -lonnxruntime `pkg-config --libs opencv4` -lboost_filesystem
OPENDR_INC = -I$(OPENDR_HOME)/include
OPENDR_LD = -L$(OPENDR_HOME)/lib -lopendr

all: download demos

demos: x3d face_recognition open_pose detr nanodet deep_sort skeleton_based_action_recognition_pst

x3d: $(BUILD_DIR)/activity_recognition/x3d_demo
face_recognition: $(BUILD_DIR)/face_recognition_demo
open_pose: $(BUILD_DIR)/lightweight_open_pose_demo
detr: $(BUILD_DIR)/object_detection_2d/detr_demo
nanodet: $(BUILD_DIR)/object_detection_2d/nanodet_jit_demo
deep_sort: $(BUILD_DIR)/object_tracking_2d/deep_sort_demo
skeleton_based_action_recognition_pst: $(BUILD_DIR)/skeleton_based_action_recognition/pst_demo
face: $(BUILD_DIR)/face_recognition_demo

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
   		$(WGET) ftp://opendrdata.csd.auth.gr/perception/pose_estimation/lightweight_open_pose/dataset/image/*; \
   		$(MV) opendrdata.csd.auth.gr/perception/pose_estimation/lightweight_open_pose/dataset/image  $(DATA_DIR)/lightweight_open_pose/database; \
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

$(BUILD_DIR)/activity_recognition/x3d_demo:
	$(MKDIR_P) $(BUILD_DIR)/activity_recognition
	@+echo "Building activity recognition x3d demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/activity_recognition/x3d_demo samples/activity_recognition/x3d/x3d_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/face_recognition_demo:
	$(MKDIR_P) $(BUILD_DIR)
	@+echo "Building face recognition demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/face_recognition_demo samples/face_recognition/face_recognition_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/lightweight_open_pose_demo:
	$(MKDIR_P) $(BUILD_DIR)
	@+echo "Building lightweight open pose demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/lightweigth_open_pose_demo samples/lightweight_open_pose/lightweight_open_pose_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/object_detection_2d/detr_demo:
	$(MKDIR_P) $(BUILD_DIR)/object_detection_2d
	@+echo "Building detr object detection demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/object_detection_2d/detr_demo samples/object_detection_2d/detr/detr_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/object_detection_2d/nanodet_jit_demo:
	$(MKDIR_P) $(BUILD_DIR)/object_detection_2d
	@+echo "Building nanodet object detection demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/object_detection_2d/nanodet_jit_demo samples/object_detection_2d/nanodet/nanodet_jit_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/object_tracking_2d/deep_sort_demo:
	$(MKDIR_P) $(BUILD_DIR)/object_tracking_2d
	@+echo "Building object tracking 2d deep sort demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/object_tracking_2d/deep_sort_demo samples/object_tracking_2d/deep_sort/deep_sort_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/skeleton_based_action_recognition/pst_demo:
	$(MKDIR_P) $(BUILD_DIR)/skeleton_based_action_recognition
	@+echo "Building skeleton based action recognition progressive spatiotemporal demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/skeleton_based_action_recognition/pst_demo samples/skeleton_based_action_recognition/progressive_spatio_temporal_gcn_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

clean:
	@+echo "Cleaning C API demo binaries and temporary files..."
	@+$(RM) -rf $(BUILD_DIR)/*
	@+$(RM) -rf $(DATA_DIR)
	@+echo "Done!"
