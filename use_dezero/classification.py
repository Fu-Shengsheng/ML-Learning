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
from dezero import DataLoader

# 超参数
# 训练的 epoch 轮数
max_epoch = 300
# 每次输入的数据量
batch_size = 30
hidden_size = 10
lr = 1.0

# 读入数据，创建模型和optimizer
train_set = dezero.datasets.Spiral()
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
# 小数点向上取整
# 每个 epoch 轮内循环执行训练的次数
# 每次取 batch_size，直到把 data 中的所有数据都读取过一遍
max_iter = math.ceil(data_size / batch_size)

avg_losses, test_losses = [], []
avg_accs, test_accs = [], []

for epoch in range(max_epoch):
    # np.random.permutation 随机重新排列数据集的索引
    index = np.random.permutation(data_size)
    sum_loss, sum_acc = 0, 0

    # 从 DataLoader 实例中取出 batch_size 规格的训练数据
    for x, t in train_loader:
        print(x.shape, t.shape)
        # predict
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    # for i in range(max_iter):
    #     # 创建小批量数据，每次从 index 中读取 batch_size 规模的数据量，获取其索引
    #     # i * batch_size：这是当前小批量的起始索引。
    #     # (i + 1) * batch_size：这是当前小批量的结束索引（不包含在内）。
    #     batch_index = index[i * batch_size : (i + 1) * batch_size]
    #     # batch 是元组(x, t)组成的数组
    #     batch = [train_set[i] for i in batch_index]
    #     batch_x = np.array([example[0] for example in batch])
    #     batch_t = np.array([example[1] for example in batch])

    #     # 计算梯度，更新参数
    #     y = model(batch_x)
    #     # loss.data 是标量，表示该批量数据集当前的平均交叉熵误差
    #     loss = F.softmax_cross_entropy(y, batch_t)
    #     model.cleargrads()
    #     loss.backward()
    #     optimizer.update()

    #     sum_loss += float(loss.data) * len(batch_t)
    
    # 输出每轮训练情况
    avg_loss = sum_loss / len(train_set)
    avg_losses.append(avg_loss)
    avg_acc = sum_acc / len(train_set)
    avg_accs.append(avg_acc)
    print('epoch %d, loss %.2f, acc %.2f' % (epoch + 1, avg_loss, avg_acc))

    test_sum_loss, test_sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            # * len(t) 的作用是将当前批次的损失和准确度乘以批次的样本数，从而得到加权的总损失和总准确度。
            # 这是为了考虑到不同批次的样本数可能不同，从而确保在计算总损失和总准确度时，每个批次对结果的贡献是按照其包含的样本数进行加权的。
            # sum_loss 就表示了所有批次的损失的加权总和，sum_acc 表示了所有批次的准确度的加权总和。
            test_sum_loss += float(loss.data) * len(t)
            test_sum_acc += float(acc.data) * len(t)
    print('test loss {:.4f}, acc {:.4f}'.format(test_sum_loss / len(test_set), test_sum_acc / len(test_set)))
    test_losses.append(test_sum_loss / len(test_set))
    test_accs.append(test_sum_acc / len(test_set))

# 绘制平均损失曲线
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.plot(np.arange(len(avg_losses)), avg_losses)
# plt.show()

# 创建画板，设置子图布局为1行2列
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(avg_losses)), avg_losses, label='Train Loss')
plt.plot(np.arange(len(test_losses)), test_losses, label='Test Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制精确度曲线
plt.subplot(1, 2, 2)
plt.plot(np.arange(len(avg_accs)), avg_accs, label='Train Accuracy')
plt.plot(np.arange(len(test_accs)), test_accs, label='Test Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()

# Plot boundary area the model predict
h = 0.001
x_min, x_max = train_set.data[:, 0].min() - .1, train_set.data[:, 0].max() + .1
y_min, y_max = train_set.data[:, 1].min() - .1, train_set.data[:, 1].max() + .1
print('x_min: %f, x_max: %f, y_min: %f, y_max: %f' % (x_min, x_max, y_min, y_max))
# 通过类似广播的方式吧 xx 和 yy 都转为二维
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print('xx shape: ', xx.shape)
print('yy_shape: ', yy.shape)
# 通过拉平 xx 和 yy 并组合，生成一个矩形区域内均匀分布的点的矩阵
X = np.c_[xx.ravel(), yy.ravel()]
print(X.shape)

with dezero.no_grad():
    # 矩形区域内均匀分布的点进行预测
    # score是包含了每个点3个类别概率的张量
    score = model(X)
# 对每个样本，选择具有最高分数的类别作为预测类别。
predict_cls = np.argmax(score.data, axis=1)
# 将预测类别的结果 predict_cls 恢复为与输入网格 xx 相同的形状。这样，Z 就成为一个网格上每个点对应的预测类别。
Z = predict_cls.reshape(xx.shape)
# 使用等高线图（contourf）来绘制决策边界。
# xx 和 yy 定义了网格上的坐标，而 Z 包含了每个点的预测类别。
# 等高线图根据这些信息在平面上填充不同颜色的区域，形成决策边界的可视化效果。
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(train_set)):
    c = train_set.label[i]
    plt.scatter(train_set.data[i][0], train_set.data[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()