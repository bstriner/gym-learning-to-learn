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
from .mnist_env import MnistEnv
from ..utils.action_mapping import ActionMappingContinuous


class MnistSgdContinuousEnv(MnistEnv):
    def __init__(self):
        action_mapping = ActionMappingContinuous(1, lambda opt: (opt.lr,))
        MnistEnv.__init__(self, action_mapping=action_mapping)

    def create_optimizer(self):
        lr = 1e-3
        self.optimizer = SGD(lr=lr)
