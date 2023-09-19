import numpy as np
# from function import Function
from variable import Variable
from function import add, square

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