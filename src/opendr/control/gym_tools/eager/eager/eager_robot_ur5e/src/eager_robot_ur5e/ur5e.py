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

import gym, gym.spaces
from gym.spaces import space
from eager_core.objects import Object, Sensor, Actuator

class UR5e(Object):
    
    def __init__(self, name: str) -> None:

        sensors = [
            Sensor(None, "joint_sensors", space=gym.spaces.Box(low=-3.14, high=3.14, shape=(6,)))
        ]

        actuators = [
            Actuator(None, "joints", space=gym.spaces.Box(low=-3.14, high=3.14, shape=(6,)))
        ]


        super().__init__("eager_robot_ur5e/ur5e", name, sensors, actuators)
