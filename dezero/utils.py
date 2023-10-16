import numpy as np
from variable import Variable

# 微分法求导
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# np.ndarray 转换函数
def as_array(x):
    # 判断判断x是否为 ndarray 类型
    if np.isscalar(x):
        return np.array(x)
    return x

# 将传入的参数类型转为 Variable 实例
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
