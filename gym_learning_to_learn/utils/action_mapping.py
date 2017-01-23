import numpy as np
import keras.backend as K
from gym.spaces import Box, MultiDiscrete, DiscreteToMultiDiscrete


class ActionMapping(object):
    def __init__(self, k, get_params, action_space):
        self.k = k
        self.get_params = get_params
        self.action_space = action_space

    def step(self, optimizer, action):
        pass


class ActionMappingContinuous(ActionMapping):
    def __init__(self, k, get_params):
        bounds = 50.0
        action_space = Box(-bounds, bounds, (k,))
        ActionMapping.__init__(self, k, get_params, action_space)

    def step(self, optimizer, action):
        params = self.get_params(optimizer)
        for param, act in zip(params, action):
            scale = np.exp(act)
            p = K.get_value(param)
            K.set_value(param, p * scale)


class ActionMappingDiscrete(ActionMapping):
    def __init__(self, k, get_params, scale=0.05):
        self.scale = scale
        action_space = DiscreteToMultiDiscrete(MultiDiscrete([3 for _ in range(k)]), 'all')
        ActionMapping.__init__(self, k, get_params, action_space)

    def step(self, optimizer, action):
        action = self.action_space(action)
        params = self.get_params(optimizer)
        for param, act in zip(params, action):
            if act == 0:
                scale = 1.0
            elif act == 1:
                scale = 1.0 - self.scale
            elif act == 2:
                scale = 1.0 + self.scale
            else:
                raise ValueError("Invalid action: {}".format(act))
            p = K.get_value(param)
            K.set_value(param, p * scale)
