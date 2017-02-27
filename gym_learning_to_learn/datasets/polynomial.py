import numpy as np
from ..utils.np_utils import split_dataset
import itertools


def load_data(n_train=32 * 50, n_val=32*5, n_test=32*5):
    # input_dim = np.random.randint(5, 10)
    input_dim = 3
    max_power = 2
    dropout = 0.5
    noise_sigma = 1e-8

    total_n = n_train + n_val + n_test
    x = np.random.uniform(-1, 1, (total_n, input_dim))
    y = np.zeros((total_n,))
    for i in range(1, max_power + 1):
        for combo in itertools.combinations_with_replacement(range(input_dim), i):
            #if np.random.uniform(0.0, 1.0) > dropout:
            tmp = np.ones((total_n,))
            for c in combo:
                tmp = tmp * x[:, c]
            coeff = np.random.uniform(-1, 1)
            y += coeff * tmp

    noise = np.random.normal(0, noise_sigma, (total_n,))
    y += noise
    y = y.reshape((-1, 1))
    sd = np.std(y, axis=None)
    m = np.mean(y, axis=None)
    y = (y - m) / sd
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    return split_dataset(x, y, n_train, n_val, n_test)
