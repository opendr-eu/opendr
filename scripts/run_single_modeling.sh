#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opendr

echo CUDA_VISIBLE_DEVICES=$1 python src/opendr/perception/object_tracking_3d/single_object_tracking/vpit/modeling.py $2 "${@:5}" --gpu_capacity=$3 --total_devices=$4
CUDA_VISIBLE_DEVICES=$1 python src/opendr/perception/object_tracking_3d/single_object_tracking/vpit/modeling.py $2 "${@:5}" --gpu_capacity=$3 --total_devices=$4