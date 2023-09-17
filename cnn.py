import numpy as np
from collections import OrderedDict
from convolution import Cnvolution
from pooling import Pooling
from layers import ReLU, SoftmaxWithLoss, Affine


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_param={
        'filter_num': 32,
        'filter_size': 3,
        'stride': 1,
        'pad': 0
    }, hidden_size=128, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_stride = conv_param['stride']
        filter_pad = conv_param['pad']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 *
                            filter_pad) / filter_stride + 1
        pool_output_size = int(
            filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        # 卷积层参数初始化
        # 生成filter_num个方形的卷积核，元素符合标准正态分布
        self.params['W1'] = weight_init_std * \
            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # 全连接层参数初始化
        self.params['W2'] = weight_init_std * \
            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Cnvolution(
            self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads
