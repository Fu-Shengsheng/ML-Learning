if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import dezero
from dezero import DataLoader, datasets, MLP, optimizers
import dezero.functions as F

max_epoch = 5
batch_size = 100
hidden_size = 1000

# 数据预处理函数
def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    # 张量元素值由0~255转化到0~1之间
    x /= 255.0
    return x

train_set = datasets.MNIST(train=True, transform=f)
test_set = datasets.MNIST(train=False, transform=f)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# 抽样展示
# x, t = train_set[0]
# plt.imshow(x.reshape(28, 28), cmap='gray')
# # 不显示坐标轴和刻度
# plt.axis('off')
# plt.show()
# print('label:', t)

# 训练
# model = MLP((hidden_size, 10))
# 指定 relu 而非默认的 sigmoid 作为隐藏层激活函数
model = MLP((hidden_size, 10), activation=F.relu)
# optimizer = optimizers.SGD().setup(model)
# 指定 MomentumSGD 作为优化器
optimizer = optimizers.MomentumSGD().setup(model)

# 加载参数，实现基于预训练的继续训练
if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

avg_losses, test_avg_losses = [], []
avg_accs, test_avg_accs = [], []
for epoch in range(max_epoch):
    sum_losses = 0
    sum_accs = 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)

        model.cleargrads()

        loss.backward()
        optimizer.update()

        acc = F.accuracy(y, t)
        sum_losses += float(loss.data) * len(t)
        sum_accs += float(acc.data) * len(t)

    avg_loss = sum_losses / len(train_set)
    avg_losses.append(avg_loss)
    avg_acc = sum_accs / len(train_set)
    avg_accs.append(sum_accs / len(train_set))


    # 测试
    with dezero.no_grad():
        sum_losses = 0
        sum_accs = 0
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_losses += float(loss.data) * len(t)
            sum_accs += float(acc.data) * len(t)

        test_avg_loss = sum_losses / len(test_set)
        test_avg_losses.append(test_avg_loss)
        test_avg_acc = sum_accs / len(test_set)
        test_avg_accs.append(test_avg_acc)

    print('epoch: {}'.format(epoch + 1))
    print('train loss: {:.4f}, train accuracy: {:.4f}, test loss: {:.4f}, test accuracy: {:.4f}'.format(avg_loss, avg_acc, test_avg_loss, test_avg_acc))

model.save_weights('my_mlp.npz')

# 创建画板，设置子图布局为1行2列
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(avg_losses)), avg_losses, label='Train Loss')
plt.plot(np.arange(len(test_avg_losses)), test_avg_losses, label='Test Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制精确度曲线
plt.subplot(1, 2, 2)
plt.plot(np.arange(len(avg_accs)), avg_accs, label='Train Accuracy')
plt.plot(np.arange(len(test_avg_accs)), test_avg_accs, label='Test Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()

