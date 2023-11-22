#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
       echo "OPENDR_HOME is not defined"
       exit 1
fi

TYPE="runtime"
if [ "$#" -ge 1 ]; then
       TYPE=$1
fi

# Required to parse the dependency files
pip install ConfigParser

# Install global dependencies
python3 parse_dependencies.py $TYPE --global
if [ -f "linux_dependencies.txt" ]; then
       cat linux_dependencies.txt | xargs sudo apt-get install --yes
       rm linux_dependencies.txt
fi
if [ -f "python_prerequisites.txt" ]; then
       cat python_prerequisites.txt | while read PACKAGE; do pip install $PACKAGE; done
       rm python_prerequisites.txt
fi
if [ -f "python_dependencies.txt" ]; then
       cat python_dependencies.txt | while read PACKAGE; do pip install $PACKAGE; done
       rm python_dependencies.txt
fi

# Install the dependencies from the work packages
python3 parse_dependencies.py $TYPE
if [ -f "linux_dependencies.txt" ]; then
       cat linux_dependencies.txt | xargs sudo apt-get install --yes
       rm linux_dependencies.txt
fi
if [ -f "python_prerequisites.txt" ]; then
       cat python_prerequisites.txt | while read PACKAGE; do pip install $PACKAGE; done
       rm python_prerequisites.txt
fi
if [ -f "python_dependencies.txt" ]; then
       cat python_dependencies.txt | while read PACKAGE; do pip install $PACKAGE; done
       rm python_dependencies.txt
fi
