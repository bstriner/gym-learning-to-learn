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


class BaseEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, action_mapping):
        self._seed()
        self.viewer = None
        self.batch_size = 32
        self.optimizer = None
        self.model = None
        self.current_step = 0
        self.action_mapping = action_mapping
        self.action_space = action_mapping.action_space
        bounds = float('inf')
        self.observation_space = spaces.Box(-bounds, bounds, (4,))
        self.viewer = None
        self.best = None
        Env.__init__(self)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def create_model(self):
        pass

    def create_optimizer(self):
        pass

    def _step(self, action):
        self.action_mapping.step(self.optimizer, action)
        loss_before = self.losses(self.data_test)
        if self.best is None:
            self.best = loss_before
        self.model.fit(self.data_train[0], self.data_train[1],
                       validation_data=(self.data_val[0], self.data_val[1]),
                       nb_epoch=1, verbose=0, batch_size=self.batch_size)
        loss_after = self.losses(self.data_test)
        self.current_step += 1
        observation = self._observation()

        reward = (self.best - loss_after)
        if loss_after < self.best:
            self.best = loss_after
        done = self.current_step > self.max_steps
        # print("Step: {}".format(observation))
        return observation, reward, done, {}

    def losses(self, data):
        loss = self.model.evaluate(data[0], data[1], verbose=0, batch_size=self.batch_size)
        return loss

    def _observation(self):
        loss_train = self.losses(self.data_train)
        loss_val = self.losses(self.data_val)
        lr = K.get_value(self.optimizer.lr)
        return np.array([loss_train, loss_val, -np.log(lr), self.current_step])

    def _reset(self):
        self.create_model()
        self.current_step = 0
        self.best = None
        observation = self._observation()
        return observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'human':
            print(self._observation())
        else:
            raise NotImplementedError("mode not supported: {}".format(mode))
