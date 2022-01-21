#!/bin/sh
export OPENDR_HOME=$PWD
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3
export LD_LIBRARY_PATH=$OPENDR_HOME/lib:$LD_LIBRARY_PATH

if [[ -z "${OPENDR_DEVICE}" ]]; then
  echo "[INFO] Set available device to CPU. You can manually change this by running 'export OPENDR_DEVICE=gpu'."
  export OPENDR_DEVICE=cpu
fi

source venv/bin/activate
