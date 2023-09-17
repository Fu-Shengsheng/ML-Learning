import numpy as np
from utils import im2col, col2im


class Cnvolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        # 卷积核（滤波器）的数目、通道数、高度、宽度
        FN, C, FH, FW = self.W.shape
        # 输入数据的数目、通道数、高度、宽度
        N, C, H, W = x.shape
        # 卷积层输出形状计算
        out_h = int(1 + (H + 2 * self.pad - FH))
        out_w = int(1 + (W + 2 * self.pad - FW))

        col = im2col(x, FH, FW, self.stride, self.pad)
        # 卷积核展开
        # reshape 的第二个参数 -1 表示自动计算该维度上剩余元素
        # 例如W的原始形状为(10,3,5,5),FN为10，则经过reshape后为(10,75)
        # 最后经过转置，转换为可以与图像数据相乘的形状
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        # 将计算结果转换为（N, C, out_h, out_w）的形状
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
