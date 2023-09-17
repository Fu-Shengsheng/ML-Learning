import numpy as np
from function import Function

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

# 定义可供直接调用的函数
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)