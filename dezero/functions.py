import numpy as np
from dezero import Function, as_variable
from dezero import utils

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
    
def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y ** 2)
        return gx
    
def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        gx = reshape(gy, self.x_shape)
        return gx
    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Trasnpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx
    
def transpose(x):
    return Trasnpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        # axis 表示轴的方向，指定 axis 可以实现对 Variable 实例进行指定轴方向的求和
        self.axis = axis
        # keepdims 表示输出是否应保留输入的维度数，即输入二维，则输出也以二维形式呈现
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        # TODO 补全reshape_sum_backward函数的实现
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        # TODO 补全broadcast_to函数的实现
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

# 矩阵的乘积实现
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    
def matmul(x, W):
    return MatMul()(x, W)

# 均方误差
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1
    
def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

# class Linear(Function):
#     # 由于计算过程需要保留 gW, gb 故 w 和 b 需要作为 inputs 而非实例内部属性
#     # 使用 __init__ 存储 W 和 b 的方法是错误的
#     def __init__(self, W, b):
#         self.W = W
#         self.b = b

#     def forward(self, x):
#         # 由于 forward 调用的时候会对入参进行解包，自动读取参数的 data 属性传入，故此处的 x 是 Variable类型 x 的 data 字段
#         # Function 中对 forward 计算结果 y 有要求必须是 ndarray，故此处需要取 W.data 而不是 W，否则 y 的类型就变成了 Variable
#         # 对于 ndarray 类型的 x 和 W 相乘，不能调用 matmul 而只能调用 dot
#         y = x.dot(self.W.data)
#         # 同理，为了保持 y 的 ndarray 类型，需要与 b.data 相加而不是直接 + b
#         if self.b is not None:
#             y += self.b.data
#         return y
    
#     def backward(self, gy):
#         print('----backward---')
#         x = self.inputs[0]
#         print('----x---')
#         print(type(x))
#         print(x.shape)
#         gb = None if self.b.data is None else sum_to(gy, self.b.shape)
#         gx = matmul(gy, self.W.T)
#         gW = matmul(x.T, gy)
#         # 由于计算过程需要保留 gW, gb 故 w 和 b 需要作为 inputs 而非实例内部属性
#         return gx, gW, gb
    
# def linear(x, W, b=None):
#     return Linear(W, b)(x)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)