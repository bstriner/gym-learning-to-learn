import numpy as np
from ..utils.np_utils import split_dataset


def load_data(n_train=32 * 50, n_val=32, n_test=32):
    total_n = n_train + n_val + n_test
    input_dim = np.random.randint(5, 10)
    x = np.random.normal(0, 1, (total_n, input_dim))
    y = np.zeros((total_n, 1))
    # max_power = np.random.randint(1, 4)
    for i in range(input_dim):
        # power = np.random.randint(0, max_power + 1)
        # if power > 0:
        power = 3
        coeffs = np.random.uniform(-1, 1, (power,))
        for j, c in enumerate(coeffs):
            d = c * np.power(x[:, i], j)
            y += d.reshape((-1, 1))
    noise = np.random.normal(0, 1e-4, (total_n, 1))
    y += noise
    # TODO: normalize y
    #y = 1/(1 + np.exp(-y))
    sd = np.std(y, axis=None)
    m = np.mean(y, axis=None)
    y = (y-m)/sd
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    return split_dataset(x, y, n_train, n_val, n_test)
