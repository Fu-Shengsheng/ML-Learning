if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F
from dezero import Variable


np.random.seed(0)
# 在0~1内随机取100个数据
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# # 绘制散点图
# plt.scatter(x, y, label='Data Points')
# plt.title('Scatter Plot of (x, y) Data Points')
# plt.xlabel('x')
# plt.ylabel('y')
# # 创建图例
# plt.legend()


# 权重初始化
# 定义输入层、隐藏层、输出层的维度
# 当前为一个输入对应一个输出，H为超参数
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

# 推导，正向传播
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

# 训练
for i in range(iters):
    y_pred = predict(x)
    # 计算损失
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()

    # 损失函数的反向传播，得到参数梯度
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    # 每隔 1000 次输出一次信息
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