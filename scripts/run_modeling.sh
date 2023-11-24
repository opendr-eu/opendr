#!/bin/bash

rm screenlog/screenlog.$5vpit-*

gpu_capacity=$3
gpus=$4

for i in $(seq 0 $(($gpus - 1)));
do
    for q in $(seq 0 $(($gpu_capacity - 1)));
    do
        id=$(( q*gpus + i ))
        name="$5vpit-gpu${i}-lid${q}-id${id}"
        echo $name
        screen -S ${name} -L -Logfile screenlog/screenlog.${name} -dm scripts/run_single_modeling.sh $1 $2 $3 $4 --id=${id}
    done
done