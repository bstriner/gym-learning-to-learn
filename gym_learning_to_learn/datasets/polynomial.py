import numpy as np
from ..utils.np_utils import split_dataset


def load_data(n_train=1000, n_val=100, n_test=100):
    total_n = n_train + n_val + n_test
    input_dim = np.random.randint(5, 25)
    x = (np.random.random((total_n, input_dim)) * 2.0) - 1.0
    y = np.zeros((total_n, 1))
    max_power = np.random.randint(1, 4)
    for i in range(input_dim):
        power = np.random.randint(0, max_power + 1)
        if power > 0:
            coeffs = np.random.random((power,))
            for j, c in enumerate(coeffs):
                y += c * np.power(x[:, i], j)
    return split_dataset(x, y, n_train, n_val, n_test)
