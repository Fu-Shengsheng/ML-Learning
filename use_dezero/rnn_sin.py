if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.optimizers
import dezero.datasets
import dezero.functions as F
import matplotlib.pyplot as plt
from dezero.models import simpleRNN

train_set = dezero.datasets.SinCurve(train=True)
# print(len(train_set))
# print(train_set[0])
# print(train_set[1])
# print(train_set[2])

# xs = [example[0] for example in train_set]
# ts = [example[1] for example in train_set]
# plt.plot(np.arange(len(xs)), xs, color='r', label='xs')
# plt.plot(np.arange(len(ts)), xs, color='b', label='xs')
# plt.show()


# 超参数
max_epoch = 100
hidden_size = 100
# BPTT（基于时间的反向传播）长度
bptt_length = 30

seqlen = len(train_set)

model = simpleRNN(hidden_size, 1)
# optimizer = dezero.optimizers.MomentumSGD().setup(model)
optimizer = dezero.optimizers.Adam().setup(model)
# optimizer = dezero.optimizers.SGD().setup(model)


# train
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        # 将输入数据转为 Dezero 可以处理的二阶张量
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        # 调整 Truncated BPTT 的时机
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            # 反向传播
            loss.backward()
            # 切断连接
            loss.unchain_backward()
            # 更新参数
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('loss.data:', loss.data, count)
    print('epoch %d | loss %f' % (epoch + 1, avg_loss))

# 验证
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
# 标记不同线的图例
plt.legend()
plt.show()