if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
import dezero.layers as L

np.random.seed(0)

x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 指定输出大小
l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    
    if i % 1000 == 0:
        print(loss)

# Plot
# 绘制散点图
plt.scatter(x, y, s=10)
# 设置坐标轴
plt.xlabel('x')
plt.ylabel('y')
# 生成一系列从0到1的间隔为0.01的数据点，并将其转换为列向量
# 这里 t 是一个列向量，用于生成模型的预测
# np.arange(0, 1, .01) 用于创建一个从 0 到 1（不包含1）的等间隔数组，间隔为 0.01
# [:, np.newaxis] 的目的是将这一维数组转换为列向量，作为神经网络的输入
# 在原数组的第二个维度上添加一个新的维度，从而将一维数组转换为列向量
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
# 绘制曲线图
# 使用 plt.plot 函数将输入 t 和相应的预测输出 y_pred.data 绘制成曲线图
plt.plot(t, y_pred.data, color='r')
plt.show()