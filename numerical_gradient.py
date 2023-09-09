import numpy as np


def numerical_gradient(x, func):
    delta = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        grad[idx] = (func(x + delta) - func(x - delta)) / (2 * delta)

    return grad
