if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import dezero.datasets
# x, t = dezero.datasets.get_spiral(train=True)
# print(x.shape)
# print(t.shape)

# print(x[10], t[10])
# print(x[110], t[110])
    
import math 
import numpy as np
import matplotlib.pyplot as plt
import dezero
import dezero.datasets
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# 超参数
# 训练的 epoch 轮数
max_epoch = 300
# 每次输入的数据量
batch_size = 30
hidden_size = 10
lr = 1.0

# 读入数据，创建模型和optimizer
x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
# 小数点向上取整
# 每个 epoch 轮内循环执行训练的次数
# 每次取 batch_size，直到把 data 中的所有数据都读取过一遍
max_iter = math.ceil(data_size / batch_size)

avg_losses = []

for epoch in range(max_epoch):
    # np.random.permutation 随机重新排列数据集的索引
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 创建小批量数据，每次从 index 中读取 batch_size 规模的数据量，获取其索引
        # i * batch_size：这是当前小批量的起始索引。
        # (i + 1) * batch_size：这是当前小批量的结束索引（不包含在内）。
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 计算梯度，更新参数
        y = model(batch_x)
        # loss.data 是标量，表示该批量数据集当前的平均交叉熵误差
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)
    
    # 输出每轮训练情况
    avg_loss = sum_loss / data_size
    avg_losses.append(avg_loss)
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

# 绘制平均损失曲线
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(np.arange(len(avg_losses)), avg_losses)
plt.show()

# Plot boundary area the model predict
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with dezero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()