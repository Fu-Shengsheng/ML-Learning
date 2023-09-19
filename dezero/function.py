import numpy as np
from variable import Variable
from utils import as_array

# Function基类定义
class Function:
    # call 的声明使类的实例可以当做可调用对象使用
    # *inputs 表示所有的参数一次性拿到，不限参数数量
    def __call__(self, *inputs):
        # 参数和返回值支持列表
        xs = [x.data for x in inputs]
        # 使用 * 号对 xs 进行解包
        ys = self.forward(*xs)

        # 对非元组的计算值额外处理
        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys]
        # 为列表list的每个元素添加creator信息
        for output in outputs:
            # 输出变量保存创造者信息
            output.set_creator(self)

        # 保存输入值
        self.inputs = inputs
        # 保存输出值
        self.outputs = outputs

        # 如果列表中只有一个元素，则返回第一个元素
        return outputs if len(outputs) > 1 else outputs[0]

    # 前向传播（计算结果）
    def forward(self, xs):
        # 抛出异常，表示这个方法应该通过继承实现
        raise NotImplementedError()

    # 反向传播（计算导数）
    def backward(self, gys):
        raise NotImplementedError()

# Square 类继承自 Function 类


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y


# 定义可供直接调用的函数
def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)

def add(x0, x1):
    f = Add()
    return f(x0, x1)