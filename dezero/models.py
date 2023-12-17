import dezero.functions as F
import dezero.layers as L
from dezero import Layer
from dezero import utils

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

# Multi-Layer Perceptron 全连接神经网络
class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        # 使用循环遍历神经网络中的所有层，除了最后一层（self.layers[-1]）
        # 这是因为在循环内部会应用激活函数，最后一层的输出不需要再经过激活函数
        for l in self.layers[: -1]:
            x = self.activation(l(x))
        # 回归任务的输出层，单独调用最后一层的全连接计算
        return self.layers[-1](x)