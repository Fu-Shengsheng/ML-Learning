import numpy as np
# from function import Function
from variable import Variable
from function import add, square
from config import Config

# xs = [Variable(np.array(2)), Variable(np.array(3))]
# f = Add()
# ys = f(xs)
# y = ys[0]
# print(y.data)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)

z = add(square(x0), square(x1))
z.backward()
print(z.data)
print(x0.grad, x1.grad)

x = Variable(np.array(3))
y = add(x, x)
print(y.data)
y.backward()
print(x.grad)

# x = Variable(np.array(3))
x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)

# 复杂计算图测试
x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad)

# 开启反向传播，模拟训练过程
Config.enable_backprop = True
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()

# 关闭反向传播，模拟预测过程
Config.enable_backprop = True
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))