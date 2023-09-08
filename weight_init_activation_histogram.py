import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reLU(x):
    return np.maximum(x, 0)


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num)

    z = np.dot(x, w)
    # a = sigmoid(z)
    a = reLU(z)
    activations[i] = a

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    # 绘制直方图，参数为：输入数据，等宽柱形的数目，柱形图的上下限（横坐标压缩范围）
    # 将每层的输出数据用柱形图表示，每层的柱形图包含30个等宽柱形，每层的柱形图横坐标压缩在0~1范围内
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
