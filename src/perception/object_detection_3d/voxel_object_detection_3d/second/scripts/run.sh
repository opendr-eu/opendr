#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tanet
export PYTHONPATH=$PYTHONPATH:~/TANet/pointpillars_with_TANet

python -W ignore ./pytorch/train.py "$@"