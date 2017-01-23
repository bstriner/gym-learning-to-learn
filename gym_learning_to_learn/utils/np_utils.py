import numpy as np


def split_data(x, n_train=1000, n_val=100, n_test=100):
    total = n_train + n_val + n_test
    assert (total == x.shape[0])
    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    total_n = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    assert (total == total_n)
    return x_train, x_val, x_test

def split_dataset(x, y, n_train=1000, n_val=100, n_test=100):
    x_train, x_val, x_test = split_data(x, n_train, n_val, n_test)
    y_train, y_val, y_test = split_data(y, n_train, n_val, n_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
