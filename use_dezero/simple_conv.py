if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

# 每张图有 C 个通道，则每个过滤器也应该有对应的 C 个通道，一次卷积完成后，累加生成了单通道的矩阵
N, C, H, W = 1, 5, 15, 15
# OC 是输出通道数，也可以理解为过滤器（卷积核）的数目
# 每个卷积核负责生成一个输出通道的特征图，因此输出通道的数量决定了卷积层输出的特征图数量。
OC, (KH, KW) = 8, (3, 3)

x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()

print(y.shape)
print(x.grad.shape)