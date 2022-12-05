#!/bin/sh
export OPENDR_HOME=$PWD
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3
export LD_LIBRARY_PATH=$OPENDR_HOME/lib:$LD_LIBRARY_PATH

source venv/bin/activate
