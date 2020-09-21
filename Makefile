# Copyright 2020 OpenDR Project
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
export WEBOTS_HOME:=`pwd -W | tr -s / '\\'`
else
export OPENDR_HOME = $(PWD)
endif
endif

ifeq ($(MAKECMDGOALS),)
MAKECMDGOALS = release
endif

.PHONY: clean release debug clean-docs docs

release debug: docs

clean: clean-docs

docs:
ifneq (, $(shell which python 2> /dev/null))
	@+echo "#"; echo "# * documentation *";
	-@+python docs/local_exporter.py --silent
else
	@+echo "#"; echo -e "# \033[0;33mPython not installed, skipping documentation\033[0m";
endif

clean-docs:
	@+echo "#"; echo "# * documentation *";
	@rm -fr docs/index.html docs/dependencies

help:
	@+echo
	@+echo -e " \033[32;1mOpenDR Toolkit Makefile targets:\033[0m"
	@+echo
	@+echo -e "\033[33;1mmake release\033[0m\t\t# compile"
	@+echo -e "\033[33;1mmake debug\033[0m\t\t# compile with gdb debugging symbols"
	@+echo -e "\033[33;1mmake clean\033[0m\t\t# clean-up the compilation output"
	@+echo -e "\033[33;1mmake docs\033[0m\t\t# install documentation dependencies"
	@+echo -e "\033[33;1mmake clean-docs\033[0m\t# cleanup documentation dependencies"
	@+echo -e "\033[33;1mmake help\033[0m\t\t# display this message and exit"
