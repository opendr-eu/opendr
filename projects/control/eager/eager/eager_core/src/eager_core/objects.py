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
from eager_core.utils.file_utils import load_yaml, substitute_xml_args
from eager_core.utils.gym_utils import (get_space_from_space_msg, get_message_from_space,
                                        get_def_from_space, get_response_from_space,
                                        get_space_from_def)
from eager_core.srv import RegisterActionProcessor, RegisterActionProcessorRequest
from eager_core.msg import Object as Object_msg
from collections import OrderedDict
from typing import Dict, List, Callable, Union
import rospy
import roslaunch
import numpy as np

class BaseRosObject():

    def __init__(self, type: str, name: str, **kwargs) -> None:
        self.type = type
        self.name = name
        self.args = str(kwargs)

    def _infer_space(self, base_topic: str = '') -> gym.Space:
        pass

    def init_node(self, base_topic: str = '') -> None:
        raise NotImplementedError
    
    def get_topic(self, base_topic: str = '') -> str:
        if base_topic == '':
            return self.name
        return base_topic + '/' + self.name


class Sensor(BaseRosObject):
    def __init__(self, type: str, name: str, space: gym.Space = None) -> None:
        super().__init__(type, name)
        self.observation_space = space

    def init_node(self, base_topic: str = '') -> None:
        if self.observation_space is None:
            self.observation_space = self._infer_space(base_topic)
        
        self._get_sensor_service = rospy.ServiceProxy(self.get_topic(base_topic), get_message_from_space(self.observation_space))

    def get_obs(self) -> object: #Type depends on space
        response = self._get_sensor_service()
        if self.observation_space.dtype == np.dtype(np.uint8):
            buf = np.frombuffer(response.value, np.uint8).reshape(self.observation_space.shape)
            return buf
        return np.array(response.value, dtype=self.observation_space.dtype).reshape(self.observation_space.shape)

    def add_preprocess(self, processed_space: gym.Space = None, launch_path='/path/to/custom/sensor_preprocess/ros_launchfile', node_type='service', stateless=True):
        self.observation_space = processed_space
        self.launch_path = launch_path
        self.node_type = node_type
        self.stateless = stateless
    
    def close(self):
        self._get_sensor_service.close()


class State(BaseRosObject):
    def __init__(self, type: str, name: str, space: gym.Space = None) -> None:
        super().__init__(type, name)
        self.state_space = space

    def init_node(self, base_topic: str = '') -> None:
        if self.state_space is None:
            self.state_space = self._infer_space(base_topic)

        self._buffer = self.state_space.sample()

        self._get_state_service = rospy.ServiceProxy(self.get_topic(base_topic),
                                                 get_message_from_space(self.state_space))

        self._send_state_service = rospy.Service(self.get_topic(base_topic), get_message_from_space(self.state_space),
                                                  self._send_state)

    def get_state(self) -> object:  # Type depends on space
        response = self._get_state_service()
        if self.state_space.dtype == np.dtype(np.uint8):
            buf = np.frombuffer(response.value, np.uint8).reshape(self.state_space.shape)
            return buf
        return np.array(response.value, dtype=self.state_space.dtype).reshape(self.state_space.shape)

    def set_state(self, state: object) -> None:
        self._buffer = state

    def _send_state(self, request: object) -> object:
        msg_class = get_response_from_space(self.state_space)
        return msg_class(self._buffer)
    
    def close(self):
        self._get_state_service.close()
        self._send_state_service.shutdown()


class Actuator(BaseRosObject):

    def __init__(self, type: str, name: str, space: gym.Space = None) -> None:
        super().__init__(type, name)
        self.action_space = space
        self.preprocess_launch = None
        self.processor_action_space = None

    def init_node(self, base_topic: str = ''):
        if self.action_space is None:
            self.action_space = self._infer_space(base_topic)

        if self.preprocess_launch is not None:
            # Launch processor
            cli_args = self.preprocess_launch
            cli_args.append('ns:={}'.format(self.get_topic(base_topic)))
            roslaunch_args = cli_args[1:]
            roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
            launch.start()
            
            # Register processor
            self._register_action_processor_service = rospy.ServiceProxy(self.get_topic(base_topic) + "/register_processor", RegisterActionProcessor)
            self._register_action_processor_service.wait_for_service()
            response = self._register_action_processor_service(self.preprocess_req)
            
            if self.processor_action_space is not None:
                self.action_space = self.processor_action_space
            else:
                self.action_space = get_space_from_space_msg(response.action_space)
            
            # Initialize buffer
            self._buffer = self.action_space.sample()
            
            # Create act service
            self._act_service = rospy.Service(self.get_topic(base_topic) + "/raw", get_message_from_space(self.action_space), self._send_action)
        else:
            self._buffer = self.action_space.sample()
            
            self._act_service = rospy.Service(self.get_topic(base_topic), get_message_from_space(self.action_space), self._send_action)
    
    def set_action(self, action: object) -> None:
        self._buffer = action

    def add_preprocess(self, launch_path: str, launch_args: dict={}, observations_from_objects: list=[], action_space: gym.Space = None):
        
        cli_args = [substitute_xml_args(launch_path)]
        for key, value in launch_args.items():
            cli_args.append('{}:={}'.format(key, value))
        self.preprocess_launch = cli_args
        
        observation_objects = []
        for object in observations_from_objects:
            object_msg = Object_msg()
            object_msg.type = object.type
            object_msg.name = object.name
            observation_objects.append(object_msg)
        
        req = RegisterActionProcessorRequest()
        if action_space is not None:
            req.raw_action_type = get_def_from_space(action_space)['type']
        req.action_type = get_def_from_space(self.action_space)['type']
        req.observation_objects = observation_objects
        self.preprocess_req = req
        
        if action_space is not None:
            self.processor_action_space = action_space
        
    def reset(self) -> None:
        pass

    def _send_action(self, request: object) -> object:
        msg_class = get_response_from_space(self.action_space)
        return msg_class(self._buffer)

    def close(self):
        self._act_service.shutdown()


class Object(BaseRosObject):
    def __init__(self, type: str, name: str, 
                 sensors: Union[List[Sensor], Dict[str, Sensor]],
                 actuators: Union[List[Actuator], Dict[str, Actuator]], 
                 states: Union[List[State], Dict[str, State]],
                 reset: Callable[['Object'], None] = None,
                 position: List[float] = [0, 0, 0],
                 orientation: List[float] = [0, 0, 0, 1],
                 fixed_base: bool = True,
                 self_collision: bool = True,
                 **kwargs
                 ) -> None:
        super().__init__(type, name, position=position, orientation=orientation, fixed_base=fixed_base, self_collision=self_collision, **kwargs)

        if isinstance(sensors, List):
            sensors = OrderedDict(zip([sensor.name for sensor in sensors], sensors))
        self.sensors = sensors if isinstance(sensors, OrderedDict) else OrderedDict(sensors)

        if isinstance(actuators, List):
            actuators = OrderedDict(zip([actuator.name for actuator in actuators], actuators))
        self.actuators = actuators if isinstance(actuators, OrderedDict) else OrderedDict(actuators)

        if isinstance(states, List):
            states = OrderedDict(zip([state.name for state in states], states))
        self.states = states if isinstance(states, OrderedDict) else OrderedDict(states)

        self.reset_func = reset
    
    @classmethod
    def create(cls, name: str, package_name: str, object_type: str,
               position: List[float] = [0, 0, 0],
               orientation: List[float] = [0, 0, 0, 1],
               fixed_base: bool = True,
               self_collision: bool = True,
               **kwargs
               ) -> 'Object':

        params = load_yaml(package_name, object_type)

        sensors = []
        if 'sensors' in params:
            for sens_name, sensor in params['sensors'].items():
                sensor_space = get_space_from_def(sensor)
                sensors.append(Sensor(None, sens_name, sensor_space))

        actuators = []
        if 'actuators' in params:
            for act_name, actuator in params['actuators'].items():
                act_space = get_space_from_def(actuator)
                actuators.append(Actuator(None, act_name, act_space))

        states = []
        if 'states' in params:
            for state_name, state in params['states'].items():
                state_space = get_space_from_def(state)
                states.append(State(None, state_name, state_space))

        return cls(package_name + '/' + object_type, name, sensors, actuators, states,
            position=position, orientation=orientation, fixed_base=fixed_base, self_collision=self_collision, **kwargs)

    def init_node(self, base_topic: str = '') -> None:

        for sensor in self.sensors.values():
            sensor.init_node(self.get_topic(base_topic))
        
        for actuator in self.actuators.values():
            actuator.init_node(self.get_topic(base_topic))

        for state in self.states.values():
            state.init_node(self.get_topic(base_topic))

    def set_action(self, actions: 'OrderedDict[str, object]') -> None: # Error return?

        for act_name, actuator in self.actuators.items():
            actuator.set_action(actions[act_name])
    
    def get_obs(self, sensors: List[str] = None) -> 'OrderedDict[str, object]':
        if sensors is not None:
            obs = OrderedDict([(sensor, self.sensors[sensor].get_obs()) for sensor in sensors])
        else:
            obs = OrderedDict([(sens_name, sensor.get_obs()) for sens_name, sensor in self.sensors.items()])
        return obs

    def get_state(self, states: List[str] = None) -> 'OrderedDict[str, object]':
        if states is not None:
            obs = OrderedDict([(state, self.states[state].get_state()) for state in states])
        else:
            obs = OrderedDict([(state_name, state.get_state()) for state_name, state in self.states.items()])
        return obs
    
    @property
    def action_space(self) -> gym.spaces.Dict:
        if not self.actuators:
            return None
        spaces = OrderedDict()
        for act_name, actuator in self.actuators.items():
            spaces[act_name] = actuator.action_space

        return gym.spaces.Dict(spaces=spaces)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        if not self.sensors:
            return None
        spaces = OrderedDict()
        for sens_name, sensor in self.sensors.items():
            spaces[sens_name] = sensor.observation_space

        return gym.spaces.Dict(spaces=spaces)

    @property
    def state_space(self) -> gym.spaces.Dict:
        if not self.states:
            return None
        spaces = OrderedDict()
        for state_name, state in self.states.items():
            spaces[state_name] = state.state_space

        return gym.spaces.Dict(spaces=spaces)

    def reset(self, states: 'OrderedDict[str, object]') -> None: # Error return?
        for actuator in self.actuators.values():
            actuator.reset()

        for state_name, state in self.states.items():
            state.set_state(states[state_name])

        if self.reset_func is not None:
            self.reset_func(self)
    
    def close(self):
        for sensor in self.sensors.values():
            sensor.close()
        for actuator in self.actuators.values():
            actuator.close()
        for state in self.states.values():
            state.close()
