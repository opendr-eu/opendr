# Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from eager_core.engine_params import EngineParams

class SafeActionsProcessor(EngineParams):
    def __init__(self,
                 moveit_package: str,
                 urdf_path: str,
                 joint_names: list,
                 robot_type: str,
                 group_name: str,
                 object_frame: str,
                 checks_per_rad: int,
                 duration: float,
                 vel_limit: float):
        kwargs = locals().copy()
        kwargs.pop('self')
        self.launch_args = kwargs
        super(SafeActionsProcessor, self).__init__(**kwargs)
