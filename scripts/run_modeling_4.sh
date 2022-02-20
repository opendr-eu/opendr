#!/bin/bash

rm screenlog/screenlog.$3vpit-*

gpu_capacity=4
gpus=4

for i in $(seq 0 $(($gpus - 1)));
do
    for q in $(seq 0 $(($gpu_capacity - 1)));
    do
        id=$(( i*gpu_capacity + q ))
        name="$3vpit-gpu${i}-lid${q}-id${id}"
        echo $name
        screen -S ${name} -L -Logfile screenlog/screenlog.${name} -dm scripts/run_single_modeling_4.sh $1 $2 --id=${id}
    done
done