#!/bin/bash

if [[ $EUID -ne 0 ]]; then
       echo "This script must be run as root"
       exit 1
fi

python runtime_dependencies.py
pip install -r python_dependencies.txt
