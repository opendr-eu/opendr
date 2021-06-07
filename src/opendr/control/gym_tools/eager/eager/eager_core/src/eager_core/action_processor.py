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

import abc
import rospy
from eager_core.srv import RegisterActionProcessor, ResetEnv
from eager_core.utils.file_utils import load_yaml
from eager_core.utils.message_utils import get_message_from_def, get_response_from_def
from eager_core.utils.gym_utils import get_message_from_space, get_space_msg_from_space
from eager_core.msg import Space

# Abstract Base Class compatible with Python 2 and 3
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()}) 

class ActionProcessor(ABC):
    
    def __init__(self):
        rospy.logdebug("[{}] Initializing processor".format(rospy.get_name()))
        self._get_observation_services = {}        
        self.__register_service = rospy.Service('register_processor', RegisterActionProcessor, self.__register_handler)
        self.__reset_service = rospy.Service('reset_processor', ResetEnv, self.__reset_handler)

    @abc.abstractmethod
    def _process_action(self, action, observation):
        return action

    @abc.abstractmethod
    def _reset(self):
        pass

    @abc.abstractmethod
    def _close(self):
        pass
    
    @abc.abstractmethod
    def _get_space(self):
        space = None
        return space
        
    def __register_handler(self, req):
        rospy.logdebug("[{}] Handling register request".format(rospy.get_name()))
        
        # Get the name of the environment
        ns = rospy.get_namespace()
        env = ns[:ns.find('/',1)+1]
        
        # The environment action space can change because of the processor
        # There are two ways to define the new action space:
        # (1) As an argument in actuator.add_preprocess()
        # (2) By implementing the self._get_space() method
        # (1) overrules (2)
        space = self._get_space()
        space_msg = Space()
        if req.raw_action_type:
            raw_action_object = {'type' : req.raw_action_type}
            raw_action_msg = get_message_from_def(raw_action_object)
        else:
            if space is None:
                rospy.logerr('[{}] Action space of the processor is unknown!'.format(rospy.get_name()))
            raw_action_msg = get_message_from_space(space)
            space_msg = get_space_msg_from_space(space)
        
        action_object = {'type' : req.action_type}
        action_msg = get_message_from_def(action_object)
        self.response = get_response_from_def(action_object)
        
        for idx, observation_object in enumerate(req.observation_objects):
            object_name = observation_object.name
            object_type = observation_object.type.split('/')
            object_params = load_yaml(object_type[0], object_type[1])
            self._get_observation_services[object_name] = {}
            for sensor in object_params['sensors']:
                sens_def = object_params['sensors'][sensor]
                msg_type = get_message_from_def(sens_def)
                self._get_observation_services[object_name][sensor] = rospy.ServiceProxy(env + 'objects/' + object_name + '/' + sensor, msg_type)
        self._get_action_service = rospy.ServiceProxy(ns + '/raw', raw_action_msg)
        self.__process_action_service = rospy.Service(ns, action_msg, self.__process_action_handler)
        return space_msg # The new environment action space
        
    def __process_action_handler(self, req):
        rospy.logdebug("[{}] Handling process request".format(rospy.get_name()))
        observation = {}
        action = self._get_action_service()
        action = action.value
        for robot in self._get_observation_services:
            observation[robot] = {}
            for sensor in self._get_observation_services[robot]:
                get_observation_service = self._get_observation_services[robot][sensor]
                obs = get_observation_service()
                observation[robot][sensor] = obs.value
        action_processed = self._process_action(action, observation)
        return self.response(action_processed)

    def __reset_handler(self, req):
        if self._reset():
            return () # Success
        else:
            return None # Error

    def __close_handler(self, req):
        if self._close():
            return () # Success
        else:
            return None # Error
