import numpy as np
# from function import Function
from variable import Variable
from function import Square, Exp, square, exp

x = Variable(np.array(10))
f = Square()
# 相当于执行了 __call__
y = f(x)
y2 = f.__call__(x)
# 获取y的类型
print(type(y), type(y2))
print(y.data, y2.data)

A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)

# 断言，如果断言不为 true，则抛出异常
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
# 省去后续的断言


y.grad = np.array(1.0)
# 获取创造者函数
C = y.creator
# 获取函数的输入
b = C.input
# 调用函数的backward
b.grad = C.backward(y.grad)

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad)


x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 反向传播
y.grad = np.array(1.0)
# 自动反向传播至原始输入变量的导数求出
y.backward()
print(x.grad)

x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

# 反向传播
y.grad = np.array(1.0)
# 自动反向传播至原始输入变量的导数求出
y.backward()
print(x.grad)

x = Variable(np.array(0.5))
y = square(exp(square(x)))
# y.grad = np.array(1.0)
y.backward()
print(x.grad)

# make type error
# x = Variable(1.0)
