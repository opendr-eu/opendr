#! /bin/bash
###python create_data.py create_kitti_info_file --data_path=/data2/zheliu/Kitti/object
###python create_data.py create_reduced_point_cloud --data_path=/data2/zheliu/Kitti/object
###python create_data.py create_groundtruth_database --data_path=/data2/zheliu/Kitti/object

CUDA_VISIBLE_DEVICES=1 python ./pytorch/train.py train --config_path=./configs/pointpillars/ped_cycle/xyres_16.proto --model_dir=/mnt/data2/TANet/pointpillars/second/train_16_ped_cyc_pointpillars