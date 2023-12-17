if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import MLP
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000

model = MLP((10, 10, 1))

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    model.cleargrads()

    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if(i % 1000 == 0):
        print(loss)

# s=10 表示散点的大小（尺寸）
# 在这里，所有的散点都被设定为相同的大小，即 10
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
pred = model(t)
plt.plot(t, pred.data, color='g')
plt.show()
