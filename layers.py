import numpy as np


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # define mask rule
        self.mask = (x <= 0)
        # x as numpy array
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        out = self.out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx
