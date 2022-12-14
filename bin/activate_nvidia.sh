#!/bin/sh
export OPENDR_HOME=$PWD
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
alias python=python3
export LD_LIBRARY_PATH=$OPENDR_HOME/lib:$LD_LIBRARY_PATH

export PATH=/usr/local/cuda/bin:$PATH
export MXNET_HOME=$OPENDR_HOME/mxnet/
export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export LC_ALL="C.UTF-8"
export MPLBACKEND=TkAgg
