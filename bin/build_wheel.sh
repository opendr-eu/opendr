#!/bin/bash


# Activate OpenDR
source ./bin/activate.sh
export OPENDR_HOME=$(pwd)


rm dist/*
rm src/*egg-info -rf

# Remove detectron and git repositories (installation not supported through PyPI)
#sed -i '/detectron2/d' requirements.txt
#sed -i '/git/d' requirements.txt
#sed -i '/pkg_resources/d' requirements.txt
#sed -i '/auditwheel/d' requirements.txt

# Build OpenDR packages
while read p; do
  echo "Build wheel for $p"
  cd src/opendr/$p
  cd $OPENDR_HOME
done < packages.txt

#cd src/opendr/engine
#python3 setup.py sdist --dist-dir=$OPENDR_HOME/dist
#cd $OPENDR_HOME
#
## Build hyperparameter tuner package
#cd src/opendr/utils/hyperparameter_tuner
#python3 setup.py sdist --dist-dir=$OPENDR_HOME/dist
#cd $OPENDR_HOME
#
#
#pip install dist/opendr-toolkit-engine-1.0.tar.gz
#pip install dist/opendr-toolkit-hyperparameter-tuner-1.0.tar.gz