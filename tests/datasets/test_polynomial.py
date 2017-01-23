from gym_learning_to_learn.datasets import polynomial
import pytest


def test_load_data():
    data = polynomial.load_data()
    assert (len(data) == 3)


if __name__ == '__main__':
    exit(pytest.main(__file__))
