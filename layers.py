from loss_func import cross_entropy_error
import numpy as np
import os
import sys
sys.path.append(os.pardir)

# ReLU 的正向、反向传播


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

# Sigmoid 的正向、反向传播


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        out = self.out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx

# Affine（衍射层）的正向、反向传播


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


def softmax(x):
    # 批处理
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    c = np.max(x)
    # 防止数据溢出
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

# 以交叉熵误差为损失函数的 Softmax 的正向、反向传播


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size

        return dx
