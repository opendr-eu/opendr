#!/bin/bash

# Fetch any submodules missing
git submodule update --init --recursive

# Clean existing packages
rm dist/*
rm src/*egg-info -rf

python3 -m pip install cython numpy

# Build OpenDR packages
while read p; do
  echo "Building wheel for $p"
  echo "exec(open('src/opendr/_setup.py').read())" > setup.py
  echo "build_package('$p')" >> setup.py
  python3 setup.py sdist
done < packages.txt

# Cleanup
rm src/*egg-info -rf
rm setup.py
rm MANIFEST.in
