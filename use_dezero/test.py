# 在使用命令行 python 解释器时，变量 __file__ 会被定义为全局变量
# 此时获取当前文件的路径，并将其父目录添加到模块的搜索路径中
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = (x + 3) * 2
y.backward()

print(y)
print(x.grad)

x = Variable(np.array(3.0))
y = (x + 1) ** 3
y.backward()

print(y)
print(x.grad)

x = Variable(np.random.rand(2, 3))
y = x.transpose()
xt = x.T
print(x, y, xt)