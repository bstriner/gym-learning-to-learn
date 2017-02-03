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
from ..datasets import polynomial
from .base_env import BaseEnv
from keras.regularizers import l1l2

class PolynomialEnv(BaseEnv):
    def __init__(self, action_mapping):
        self.output_dim = 1
        self.batch_size = 32
        self.max_steps = 200
        self.data_train, self.data_val, self.data_test = None, None, None
        BaseEnv.__init__(self, action_mapping=action_mapping)

    def create_model(self):
        self.data_train, self.data_val, self.data_test = polynomial.load_data()
        input_dim = self.data_train[0].shape[1]
        x = Input((input_dim,))
        reg = lambda: l1l2(1e-7, 1e-7)
        h = Dense(128, activation='tanh', W_regularizer=reg())(x)
        h = Dense(64, activation='tanh', W_regularizer=reg())(h)
        h = Dense(32, activation='tanh', W_regularizer=reg())(h)
        y = Dense(self.output_dim, W_regularizer=reg())(h)
        self.model = Model(x, y)
        self.create_optimizer()
        self.model.compile(self.optimizer, 'mean_squared_error')

