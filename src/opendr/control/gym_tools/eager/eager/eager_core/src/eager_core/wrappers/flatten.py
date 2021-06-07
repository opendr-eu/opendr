import gym, gym.spaces
from gym.core import ActionWrapper
from gym.wrappers import FlattenObservation


class FlattenAction(ActionWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env: gym.Env) -> None:
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.flatten_space(env.action_space)

    def action(self, action: object) -> object:
        return gym.spaces.unflatten(self.env.action_space, action)


class Flatten(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(FlattenObservation(FlattenAction(env)))
