# Copyright 2020-2021 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ifeq ($(OPENDR_HOME),)
ifneq ($(findstring MINGW,$(shell uname)),) # under MINGW, we need to set OPENDR_HOME using the native Windows format
export OPENDR_HOME:=`pwd -W | tr -s / '\\'`
else
export OPENDR_HOME = $(PWD)
endif
endif

ifeq ($(MAKECMDGOALS),)
MAKECMDGOALS = release
endif

.PHONY: release install_compilation_dependencies install_runtime_dependencies

release: install_compilation_dependencies

install_runtime_dependencies:
	@+echo "#"; echo "# * Install Runtime Dependencies *"; echo "#"
	@+cd dependencies; ./install.sh runtime
	@+cd src/opendr/perception/object_detection_2d/retinaface; make

install_compilation_dependencies:
	@+echo "#"; echo "# * Install Compilation Dependencies *"; echo "#"
	@+cd dependencies; ./install.sh compilation
	@+cd dependencies; ./install_onnx.sh
	@+make --silent -C src/opendr/control/mobile_manipulation $(TARGET) OPENDR_HOME="$(OPENDR_HOME)";
	@+make --silent -C src/opendr/control/single_demo_grasp $(TARGET) OPENDR_HOME="$(OPENDR_HOME)";

styletest:
	@+echo "Testing file licences and code-style"
	@+python3 -m pip install -r tests/requirements.txt
	@+python3 -m unittest discover -s tests

unittest:
	@+echo "Performing unit tests"
	@+python3 -m pip install -r tests/sources/requirements.txt
	@+python3 -m unittest discover -s tests/sources/tools/

libopendr:
	@$(MAKE) -C src/c_api all

ctests: libopendr
	@$(MAKE) -C tests runtests
	@$(MAKE) -C tests clean


clean:
	@$(MAKE) -C src/c_api clean
	@$(MAKE) -C tests clean

help:
	@+echo
	@+echo -e "\033[32;1mOpenDR Makefile targets:\033[0m"
	@+echo
	@+echo -e "\033[33;1mmake -j$(THREADS) release\033[0m\t# install dependencies and compile (default)"
	@+echo -e "\033[33;1mmake -j$(THREADS) dependencies\033[0m\t# install toolkit dependencies"
	@+echo -e "\033[33;1mmake help\033[0m\t\t# display this message and exit"
	@+echo -e "\033[33;1mmake styletest\033[0m\t# run tests for style and licences"
	@+echo -e "\033[33;1mmake unittest\033[0m\t# run unit tests"
	@+echo -e "\033[33;1mmake libopendr\033[0m\t# builds the OpenDR C API"
	@+echo -e "\033[33;1mmake clean\033[0m\t\t# cleans build and temporary files"
	@+echo
	@+echo -e "\033[32;1mNote:\033[0m You seem to have a processor with $(NUMBER_OF_PROCESSORS) virtual cores,"
	@+echo -e "      hence the \033[33;1m-j$(THREADS)\033[0m option to speed-up the compilation."
