
from keras.optimizers import SGD
from .mnist_env import MnistEnv
from ..utils.action_mapping import ActionMappingContinuous


class MnistSgdContinuousEnv(MnistEnv):
    def __init__(self):
        action_mapping = ActionMappingContinuous(1, lambda opt: (opt.lr,))
        MnistEnv.__init__(self, action_mapping=action_mapping)

    def create_optimizer(self):
        lr = 1e-3
        self.optimizer = SGD(lr=lr)
