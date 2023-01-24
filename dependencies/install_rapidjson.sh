#!/bin/bash

if [ ! -d /usr/local/include/rapidjson ]; then

  VERSION="1.1.0"

  wget https://github.com/Tencent/rapidjson/archive/refs/tags/v${VERSION}.tar.gz --quiet
  tar zxf v${VERSION}.tar.gz
  cd rapidjson-${VERSION}
  sudo mkdir -p /usr/local/include/rapidjson
  sudo mv include/rapidjson/* /usr/local/include/rapidjson
  cd ..
  rm -rf rapidjson-${VERSION}
  rm -rf v${VERSION}.tar.gz


fi
