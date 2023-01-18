# Copyright 2020-2023 OpenDR European Project
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

ifeq ($(MAKECMDGOALS),)
MAKECMDGOALS = release
endif

.PHONY: release install_compilation_dependencies install_runtime_dependencies

release: install_compilation_dependencies

install_runtime_dependencies:
	@+echo "#"; echo "# * Install Runtime Dependencies *"; echo "#"

install_compilation_dependencies:
	@+echo "#"; echo "# * Install Compilation Dependencies for single demonstration grasping *"; echo "#"
	@+python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
	@./install_single_demo_grasp.sh

help:
	@+echo
	@+echo -e "\033[32;1mOpenDR Makefile targets:\033[0m"
	@+echo
	@+echo -e "\033[33;1mmake -j$(THREADS) release\033[0m\t# install dependencies and compile (default)"
	@+echo -e "\033[33;1mmake -j$(THREADS) dependencies\033[0m\t# install toolkit dependencies"
	@+echo -e "\033[33;1mmake help\033[0m\t\t# display this message and exit"
	@+echo
	@+echo -e "\033[32;1mNote:\033[0m You seem to have a processor with $(NUMBER_OF_PROCESSORS) virtual cores,"
	@+echo -e "      hence the \033[33;1m-j$(THREADS)\033[0m option to speed-up the compilation."
