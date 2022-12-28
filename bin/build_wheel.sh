#!/bin/bash

# Fetch any submodules missing
git submodule update --init --recursive

# Clean existing packages
rm dist/*
rm src/*egg-info -rf

pip install cython numpy<=1.23.5

# Build OpenDR packages
while read p; do
  echo "Building wheel for $p"
  echo "exec(open('src/opendr/_setup.py').read())" > setup.py
  echo "build_package('$p')" >> setup.py
  python3 setup.py sdist
done < packages.txt

# Cleanup
rm -rf src/*.egg-info
rm setup.py
rm MANIFEST.in
