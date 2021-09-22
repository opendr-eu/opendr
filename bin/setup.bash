#!/bin/sh
export OPENDR_HOME=$PWD
export OPENDR_DEVICE=gpu
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3

if [[ ! -d "venv" ]]; then
	pip3 install virtualenv
	virtualenv -p python3 venv
fi
source venv/bin/activate
python -m pip install -U pip

echo "[INFO] Set available device to GPU. You can manually change this by running 'export OPENDR_DEVICE=cpu'."
