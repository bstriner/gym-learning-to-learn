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
        space = MultiDiscrete([[0, 2] for _ in range(k)])
        action_space = DiscreteToMultiDiscrete(space, 'all')
        ActionMapping.__init__(self, k, get_params, action_space)

    def step(self, optimizer, action):
        action = self.action_space(action)
        params = self.get_params(optimizer)
        for param, act in zip(params, action):
            mul = 1.0 + self.scale
            if act == 0:
                scale = 1.0/mul
            elif act == 1:
                scale = 1.0
            elif act == 2:
                scale = mul
            else:
                raise ValueError("Invalid action: {}".format(act))
            p = K.get_value(param)
            p = p * scale
            p = min(p, 1e-1)
            p = max(p, 1e-9)
            #if p > 0.01:
            #    p = 0.01
            #K.set_value(param, np.float32(p))
            #print("LR update: {} -> {}".format(p, p * scale))
