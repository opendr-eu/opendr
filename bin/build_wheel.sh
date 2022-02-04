#!/bin/bash

# Fetch any submodules missing
git submodule update --init --recursive

# Clean existing packages
rm dist/*
rm src/*egg-info -rf

# Build OpenDR packages
while read p; do
  echo "Building wheel for $p"
  echo "exec(open('src/opendr/_setup.py').read())" > setup.py
  echo "build_package('$p')" >> setup.py
  python3 setup.py sdist
done < packages.txt

rm setup.py
rm MANIFEST.in
