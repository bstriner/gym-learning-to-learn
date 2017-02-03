from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
from string import Template
import os
import sys
import numpy as np
import time
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import SGD
import keras.backend as K
from ..datasets import mnist
from .base_env import BaseEnv
from .polynomial_env import PolynomialEnv
from ..utils.action_mapping import ActionMappingDiscrete


class PolynomialSgdDiscreteEnv(PolynomialEnv):
    def __init__(self):
        action_mapping = ActionMappingDiscrete(1, lambda opt: (opt.lr,), scale=0.1)
        PolynomialEnv.__init__(self, action_mapping=action_mapping)

    def create_optimizer(self):
        lr = 1e-2
        self.optimizer = SGD(lr=lr)
