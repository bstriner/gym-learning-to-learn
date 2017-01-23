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


class MnistEnv(BaseEnv):
    def __init__(self, action_mapping):
        self.input_shape = (28, 28)
        self.output_dim = 10
        self.batch_size = 32
        self.data_train, self.data_val, self.data_test = mnist.load_data()
        self.max_steps = 100
        BaseEnv.__init__(self, action_mapping=action_mapping)

    def create_optimizer(self):
        self.optimizer = SGD(lr=lr)


    def create_model(self):
        x = Input(self.input_shape)
        h = Flatten()(x)
        h = Dense(256, activation='tanh')(h)
        h = Dense(128, activation='tanh')(h)
        y = Dense(self.output_dim, activation='softmax')(h)
        self.model = Model(x, y)
        lr = 1e-3
        self.create_optimizer()
        self.model.compile(self.optimizer, 'categorical_crossentropy')
        self.current_step = 0

