#!/bin/bash

# Build all OpenDR dependecies
#./bin/install.sh

# Activate OpenDR
source ./bin/activate.sh

# Prepare requirements.txt for wheel distributions
pip3 freeze > requirements.txt

# Remove detectron and git repositories (installation not supported through PyPI)
sed -i '/detectron2/d' requirements.txt
sed -i '/git/d' requirements.txt
sed -i '/pkg_resources/d' requirements.txt
sed -i '/auditwheel/d' requirements.txt

# Build binary wheel and repair it
python3 setup.py bdist_wheel
pip3 install auditwheel
auditwheel repair dist/*.whl -w dist/ --plat manylinux_2_24_x86_64
