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
from eager_core.utils.file_utils import substitute_xml_args
from scenic.simulators.webots import world_parser


class WebotsEngine(EngineParams):
    def __init__(self,
                 world: str,
                 dt: float = 0.08,
                 no_gui: str = 'false',
                 mode: str = 'fast'):
        # Only define variables (locally) you wish to store on the parameter server (done in baseclass constructor).
        bridge_type = 'webots'
        launch_file = '$(find eager_bridge_%s)/launch/%s.launch' % (bridge_type, bridge_type)

        # Store parameters as properties in baseclass
        # IMPORTANT! Do not define variables locally you do **not** want to store
        # on the parameter server anywhere before calling the baseclass' constructor.
        kwargs = locals().copy()
        kwargs.pop('self')
        super(WebotsEngine, self).__init__(**kwargs)

        # Calculate other parameters based on previously defined attributes.
        self.step_time = int(self.dt * 1000)
        # todo: check if problem that world_parser import requires python version > 3.7
        wf = world_parser.parse(substitute_xml_args('%s' % world))  # Grab basicTimeStep from world file (.wbt).
        val = world_parser.findNodeTypesIn(['WorldInfo'], wf, nodeClasses={})
        self.basicTimeStep = int(val[0][0].attrs['basicTimeStep'][0])

        # Error check the parameters here.
        if self.step_time % self.basicTimeStep != 0:
            raise RuntimeError('The steptime (%d ms) is not a multiple of the basicTimeStep (%d ms).' % (self.step_time, self.basicTimeStep))

