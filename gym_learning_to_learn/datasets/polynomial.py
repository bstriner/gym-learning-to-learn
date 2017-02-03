import numpy as np
from ..utils.np_utils import split_dataset


def load_data(n_train=32*5, n_val=32, n_test=32):
    total_n = n_train + n_val + n_test
    input_dim = np.random.randint(5, 15)
    x = (np.random.random((total_n, input_dim)) * 2.0) - 1.0
    y = np.zeros((total_n, 1))
    max_power = np.random.randint(1, 4)
    for i in range(input_dim):
        power = np.random.randint(0, max_power + 1)
        if power > 0:
            coeffs = np.random.random((power,))
            for j, c in enumerate(coeffs):
                d = c * np.power(x[:, i], j)
                y += d.reshape((-1, 1))
    noise_val = np.random.uniform(0, 1) * 1e-3
    noise = np.random.random((total_n, 1))*noise_val
    y += noise
    return split_dataset(x, y, n_train, n_val, n_test)
