import dezero.functions as F
import dezero.layers as L
import numpy as np
from dezero import Layer
from dezero import utils

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

# Multi-Layer Perceptron 全连接神经网络
class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        # 使用循环遍历神经网络中的所有层，除了最后一层（self.layers[-1]）
        # 这是因为在循环内部会应用激活函数，最后一层的输出不需要再经过激活函数
        for l in self.layers[: -1]:
            x = self.activation(l(x))
        # 回归任务的输出层，单独调用最后一层的全连接计算
        return self.layers[-1](x)
    
# VGG16的实现
class VGG16(Model):
    # 预训练数数据权重
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

    # pretrained 为 true 时，从 WEIGHTS_PATH 下载并读取权重文件
    def __init__(self, pretrained=False):
        super().__init__()
        # 只指定输出的通道数
        # 实际初始输入的通道数由输入数据张量的第二维（第一维是图片数目N）决定
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)

        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)

        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)

        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)

        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)

        # 只指定输出大小
        # 实际初始输入大小由输入数据张量的第二维（第一维是图片数目N）决定
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)

        # 最终输出的大小
        self.fc8 = L.Linear(1000)

        # 加载预训练权重
        if pretrained:
            weight_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weight_path)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling_simple(x, 2, 2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling_simple(x, 2, 2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling_simple(x, 2, 2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling_simple(x, 2, 2)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling_simple(x, 2, 2)
        
        # 转为 (N, OH * OW) 的形状，便于进行后续的全连接层计算
        x = F.reshape(x, (x.shape[0], -1))

        # linear & relu & dropout
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))

        x = self.fc8(x)
        return x

    # 声明预处理静态函数
    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        # 将输入的图像转换为RGB模式。这是因为有些图像可能是单通道（灰度图像），而深度学习模型通常要求输入是RGB格式的图像。
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        # 颜色通道顺序的调整。
        # 通常，深度学习模型训练时使用的图像数据格式是BGR，而不是常见的RGB。
        # 通过[::-1]操作将RGB顺序调整为BGR。
        image = image[:, :, ::-1]
        # 减去均值。
        # 这一行代码减去了一个均值向量，通常是在大量图像上计算得到的，目的是对图像进行零中心化处理。
        # 这有助于训练过程更稳定，避免了大的输入值对模型的影响。
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        # 调整数组的维度顺序。
        # 这一行代码将数组的维度从(H, W, C)调整为(C, H, W)，以符合深度学习框架对输入数据的要求。
        image = image.transpose((2, 0, 1))
        return image
    
class simpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y
