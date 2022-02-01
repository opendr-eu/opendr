#!/bin/bash

# Activate OpenDR
source ./bin/activate.sh
export OPENDR_HOME=$(pwd)

rm dist/*
rm src/*egg-info -rf

# Build OpenDR packages
while read p; do
  echo "Build wheel for $p"
  cd src/opendr/$p
  python3 setup.py sdist --dist-dir=$OPENDR_HOME/dist
  cd $OPENDR_HOME
done < packages.txt


pip install dist/opendr-toolkit-engine-1.0.tar.gz
pip install dist/opendr-toolkit-face-recognition-1.0.tar.gz
pip install dist/opendr-toolkit-pose-estimation-1.0.tar.gz
pip install dist/opendr-toolkit-hyperparameter-tuner-1.0.tar.gz
pip install dist/opendr-toolkit-semantic-segmentation-1.0.tar.gz
pip install dist/opendr-toolkit-speech-recognition-1.0.tar.gz
pip install dist/opendr-toolkit-compressive-learning-1.0.tar.gz
