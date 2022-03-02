#!/bin/bash

rm screenlog/screenlog.multi-eval-$3vpit-*

gpu_capacity=4
gpus=4

for i in $(seq 0 $(($gpus - 1)));
do
    for q in $(seq 0 $(($gpu_capacity - 1)));
    do
        id=$(( i*gpu_capacity + q ))
        name="multi-eval-$3vpit-gpu${i}-lid${q}-id${id}"
        echo $name
        screen -S ${name} -L -Logfile screenlog/screenlog.${name} -dm scripts/run_single_multi_eval.sh $1 --params_file=$2 --id=${id} --gpu_capacity=${gpu_capacity} --total_devices=${gpus} "${@:4}"
    done
done