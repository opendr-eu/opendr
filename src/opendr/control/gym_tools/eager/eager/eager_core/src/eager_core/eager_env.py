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
import rospy, roslaunch, rosparam
from gym.utils import seeding
from collections import OrderedDict
from typing import List, Tuple
from eager_core.objects import Object
from eager_core.srv import StepEnv, ResetEnv, Register
from eager_core.utils.file_utils import substitute_xml_args
from eager_core.msg import Seed, Object as ObjectMsg
from eager_core.engine_params import EngineParams

class BaseEagerEnv(gym.Env):
    def __init__(self, engine: EngineParams, name: str = 'ros_env') -> None:
        super().__init__()

        self._initialize_physics_bridge(engine, name)

        self.name = name

        self._step = rospy.ServiceProxy(name + '/step', StepEnv)
        self._reset = rospy.ServiceProxy(name + '/reset', ResetEnv)
        self.__seed_publisher = rospy.Publisher(name + '/seed', Seed, queue_size=1)

    def _init_nodes(self, objects: List[Object] = [], observers: List['Observer'] = []) -> None:

        self._register_objects(objects, observers)
        self._init_listeners(objects, observers)

    def _register_objects(self, objects: List[Object] = [], observers: List['Observer'] = []) -> None:
        objects_reg = []

        for el in (objects, observers):
            for object in el:
                objects_reg.append(ObjectMsg(object.type, object.name, object.args))

        register_service = rospy.ServiceProxy(self.name + '/register', Register)
        register_service.wait_for_service(200000)
        register_service(objects_reg)

    def _init_listeners(self, objects: List[Object] = [], observers: List['Observer'] = []) -> None:
        bt = self.name + '/objects'

        for el in (objects, observers):
            for object in el:
                object.init_node(bt)

    def _merge_spaces(cls, objects: List[Object] = [], observers: List['Observer'] = []) -> Tuple[gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict]:
        
        obs_spaces = OrderedDict()
        act_spaces = OrderedDict()
        state_spaces = OrderedDict()

        for object in objects:
            if object.observation_space:
                obs_spaces[object.name] = object.observation_space
            if object.action_space:
                act_spaces[object.name] = object.action_space
            if object.state_space:
                state_spaces[object.name] = object.state_space
        
        for observer in observers:
            obs_spaces[observer.name] = observer.observation_space
        
        return gym.spaces.Dict(spaces=obs_spaces), gym.spaces.Dict(spaces=act_spaces), gym.spaces.Dict(spaces=state_spaces)

    def _initialize_physics_bridge(self, engine: EngineParams, name: str):
        # Delete all parameters parameter server (from a previous run) within namespace 'name'

        # Upload dictionary with engine parameters to ROS parameter server
        rosparam.upload_params('%s/physics_bridge' % name, engine.__dict__)

        # Launch the physics bridge under the namespace 'name'
        cli_args = [substitute_xml_args(engine.launch_file),
                    'name:=' + name]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self._launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        self._launch.start()
    
    def close(self, objects=[], observers=[]):
        self._launch.shutdown()

        for el in (objects, observers):
            for object in el:
                object.close()
        
        try:
            rosparam.delete_param('/%s' % self.name)
            rospy.loginfo('Pre-existing parameters under namespace "/%s" deleted.' % self.name)
        except:
            pass
    def seed(self, seed: int = None) -> List[int]:
        # todo: does this actually seed e.g. numpy on the EagerEnv side?
        seed = seeding.create_seed(seed)
        self.__seed_publisher.publish(seed)
        return [seed]


class EagerEnv(BaseEagerEnv):

    def __init__(self, objects: List[Object] = [], observers: List['Observer'] = [], render_obs=None, max_steps=None, reward_fn=None, is_done_fn=None, **kwargs) -> None:
        # todo: Interface changes a lot, use **kwargs.
        #  Make arguments of baseclass explicit when interface is more-or-less fixed.
        super().__init__(**kwargs)
        self.objects = objects
        self.observers = observers
        self.render_obs = render_obs

        # Overwrite the _get_reward() function if provided
        if reward_fn:
            self._get_reward = reward_fn

        # Overwrite the _is_done() function if provided
        if is_done_fn:
            self._is_done = is_done_fn
        self.STEPS_PER_ROLLOUT = max_steps
        self.steps = 0

        self._init_nodes(self.objects, self.observers)

        self.observation_space, self.action_space, self.state_space = self._merge_spaces(self.objects, self.observers)
    
    def step(self, action: 'OrderedDict[str, object]') -> Tuple[object, float, bool, dict]:
        self.steps += 1

        for object in self.objects:
            if object.action_space:
                object.set_action(action[object.name])

        self._step()

        state = self._get_states()
        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = self._is_done(obs)
        return obs, reward, done, {'state': state}
    
    def reset(self) -> object:
        self.steps = 0
        for obj in self.objects:
            if obj.state_space:  # Check if object has state that we can reset
                reset_states = obj.state_space.sample()

                # currently resetting to zero state.
                for state in reset_states:
                    reset_states[state] *= 0

                # Set state we want to reset to in buffer
                obj.reset(states=reset_states)

        self._reset()

        return self._get_obs()

    def render(self, mode: str = 'human', **kwargs) -> None:
        if self.render_obs:
            obs = self.render_obs()
            return obs
        else:
            return None

    def _get_obs(self) -> 'OrderedDict[str, object]':
        obs = OrderedDict()

        for object in self.objects:
            if object.observation_space:
                obs[object.name] = object.get_obs()

        for observer in self.observers:
            obs[observer.name] = observer.get_obs()

        return obs
    
    def _get_states(self) -> 'OrderedDict[str, object]':
        state = OrderedDict()

        for object in self.objects:
            if object.state_space:
                state[object.name] = object.get_state()

        return state

    def _get_reward(self, obs: 'OrderedDict[str, object]') -> float:
        return 0.0
    
    def _is_done(self, obs: 'OrderedDict[str, object]') -> bool:
        if self.STEPS_PER_ROLLOUT and self.STEPS_PER_ROLLOUT > 0:
            return self.steps >= self.STEPS_PER_ROLLOUT
        else:
            return False

    def close(self):
        super().close(objects=self.objects, observers=self.observers)
