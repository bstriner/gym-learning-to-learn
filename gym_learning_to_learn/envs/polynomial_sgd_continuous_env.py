from keras.optimizers import SGD
from .polynomial_env import PolynomialEnv
from ..utils.action_mapping import ActionMappingContinuous


class PolynomialSgdContinuousEnv(PolynomialEnv):
    def __init__(self):
        action_mapping = ActionMappingContinuous(1, lambda opt: (opt.lr,), limits=[[1e-9, 1e-1]])
        PolynomialEnv.__init__(self, action_mapping=action_mapping)

    def create_optimizer(self):
        lr = 1e-2
        self.optimizer = SGD(lr=lr)
