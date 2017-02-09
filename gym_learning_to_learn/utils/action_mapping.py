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
    def __init__(self, k, get_params, limits, log_scale=True, scale=1.0):
        self.limits = limits
        self.log_scale = log_scale
        self.scale = scale
        bounds = 50.0
        action_space = Box(-bounds, bounds, (k,))
        ActionMapping.__init__(self, k, get_params, action_space)

    def step(self, optimizer, action):
        params = self.get_params(optimizer)
        for param, act, limit in zip(params, action, self.limits):
            p = K.get_value(param)
            if self.log_scale:
                pnext = np.clip(np.exp(np.log(p) + (act * self.scale)), limit[0], limit[1])
            else:
                pnext = np.clip(p + (act * self.scale), limit[0], limit[1])
            K.set_value(param, np.float32(pnext))


class ActionMappingDiscrete(ActionMapping):
    def __init__(self, k, get_params, limits, scale=0.05):
        self.scale = scale
        space = MultiDiscrete([[0, 2] for _ in range(k)])
        action_space = DiscreteToMultiDiscrete(space, 'all')
        self.limits = limits
        ActionMapping.__init__(self, k, get_params, action_space)

    def step(self, optimizer, action):
        action = self.action_space(action)
        params = self.get_params(optimizer)
        for param, act, limit in zip(params, action, self.limits):
            mul = 1.0 + self.scale
            if act == 0:
                scale = 1.0 / mul
            elif act == 1:
                scale = 1.0
            elif act == 2:
                scale = mul
            else:
                raise ValueError("Invalid action: {}".format(act))
            p = K.get_value(param)
            pnext = np.clip(p * scale, limit[0], limit[1])
            K.set_value(param, np.float32(pnext))
