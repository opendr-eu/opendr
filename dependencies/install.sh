#!/bin/bash

if [[ -z "$OPENDR_HOME" ]]; then
       echo "OPENDR_HOME is not defined"
       exit 1
fi

pip install ConfigParser numpy cython

python runtime_dependencies.py
# install dependencies one by one to prevent interdependency errors
cat python_dependencies.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 python -m pip install
