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

.PHONY: release debug profile clean

release debug profile clean:
	+@echo "# compile libraries"
	+@make -s -C webots/libraries/bvh_util $@
	+@echo "# compile controller"
	+@make -s -C webots/controllers/bvh_animation $@
	+@echo "# compile plugins"
	+@make -s -C webots/plugins/robot_windows/robotti_window $@
