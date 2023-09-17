from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import numpy as np
import sys
import os
sys.path.append(os.pardir)

# 读取训练和测试数据
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=64, output_size=10)

# 训练数据规模
train_size = x_train.shape[0]

# 总训练次数及批次大小
total_train_count = 10000
batch_size = 100

# 学习率
learning_rate = 0.1

# 存放训练过程数据
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 每个epoch包含的训练次数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(total_train_count):
    # 每次从 x_train 数据集中随机挑选大小为 batch_size 的一批数据进行训练
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.gradient(x_batch, t_batch)

    # 优化器，更新梯度
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 计算并记录损失数据
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 每经过一个 epoch，计算并记录当前的识别精度和全量测试数据集的测试精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch, t_batch)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(
            f'epoch index {i}, traic_acc is {train_acc}, test_acc is{test_acc}')
