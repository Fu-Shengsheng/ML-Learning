if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
import dezero.utils
from dezero.models import simpleRNN

# 生成虚拟的时间序列数据
seq_data = [np.random.randn(1, 1) for _ in range(1000)]
# xs 包含了从序列的第一个元素到倒数第二个元素（不包括最后一个元素）的所有元素
xs = seq_data[0: -1]
# ts 包含了从序列的第二个元素到最后一个元素的所有元素
ts = seq_data[1:]

model = simpleRNN(10, 1)
loss, cnt = 0, 0

# 将 xs 和 ts 的对应位置的元素配对，生成一个新的迭代器，该迭代器每次返回一个元组
for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)
    cnt += 1
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break