#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opendr

echo CUDA_VISIBLE_DEVICES=$1 python src/opendr/perception/object_tracking_3d/single_object_tracking/voxel_bof/test.py multi_eval "${@:2}"
CUDA_VISIBLE_DEVICES=$1 python src/opendr/perception/object_tracking_3d/single_object_tracking/voxel_bof/test.py multi_eval "${@:2}"