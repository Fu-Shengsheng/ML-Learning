import numpy as np
from dezero.core import Function, as_variable
from dezero.functions import linear
from dezero.utils import pair, get_conv_outsize

def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)
    # 卷积核数据
    Weight = W
    # 输入数据的形状
    N, C, H, W = x.shape
    # 卷积核形状
    OC, C, KH, KW = Weight.shape
    # 垂直和水平方向的步幅
    SH, SW = pair(stride)
    # 水平和垂直方向的 pad
    PH, PW = pair(pad)
    # 输出形状
    OH  = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # 每张图通过 im2col 展开
    # 展开过程为：
    # 1、每次取卷积核相同形状的的图片像素，展开为长度为 C*KH*KW 长度的列
    # 2、根据输出形状，最终每张图片的所有像素展开为 (OH*OW, C*KH*KW) 形状的张量
    # 3、N张图像，则最终转换为 (N*OH*OW, C*KH*KW) 形状的二维张量
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    # 将卷积核并排展开为一列，并进行转置，以满足矩阵乘法计算
    # 每个卷积核展开为 C*KH*KW 长度的列
    # OC 个卷积核展开为 (OC, C*KH*KW)
    # 转置后生成(C*KH*KW, OC)形状的张量
    Weight = Weight.reshape(OC, -1).transpose()

    # 线性计算后得到 (N*OH*OW, OC) 形状的张量
    t = linear(col, Weight, b)
    # 由于目标张量的形状为（N, OC, OH, OW）,故需要进行换轴转置
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y

def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)
    # 获取卷积处理后的矩阵各个维度的数据
    N, C, H, W = x.shape
    # 池化核的形状
    KH, KW = pair(kernel_size)
    # pad
    PH, PW = pair(pad)
    # stride
    SH, SW = pair(stride)

    # 获取输出尺寸
    # 池化和卷积的计算方式相同，通常卷积的 stride 为 1，而池化的 stride 为 kernel_size
    OH  = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # 将每张图像展开为一维数据，形状为 (C*KH*KW, OH, OW)
    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    # 每张图 reshape 形状为 (C*OH*OW, KH*KW)
    # 此时每行就是待池化的数据，大小与池化核相同
    col = col.reshape(-1, KH * KW)
    # 求每行的最大值，得到 (C*KH*KW, 1) 形状的张量
    # N 张图张量形状为 (N, C*KH*KW, 1)
    y = col.max(axis=1)
    # 每张图 reshape 的形状为 (OH, OW, C)
    # 得到的 y 形状为 (N, OH, OW, C)
    # 因为输出的目标张量的颜色通道在第二维，故需要进行换轴转置
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y

# =============================================================================
#  im2col / col2im
# =============================================================================
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
                    self.pad, self.to_matrix)
        return gx


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    """Extract patches from an image based on the filter.

    Args:
        x (`dezero.Variable` or `ndarray`): Input variable of shape
            `(N, C, H, W)`
        kernel_size (int or (int, int)): Size of kernel.
        stride (int or (int, int)): Stride of kernel.
        pad (int or (int, int)): Spatial padding width for input arrays.
        to_matrix (bool): If True the `col` will be reshaped to 2d array whose
            shape is `(N*OH*OW, C*KH*KW)`

    Returns:
        `dezero.Variable`: Output variable. If the `to_matrix` is False, the
            output shape is `(N, C, KH, KW, OH, OW)`, otherwise
            `(N*OH*OW, C*KH*KW)`.

    Notation:
    - `N` is the batch size.
    - `C` is the number of the input channels.
    - `H` and `W` are the height and width of the input image, respectively.
    - `KH` and `KW` are the height and width of the filters, respectively.
    - `SH` and `SW` are the strides of the filter.
    - `PH` and `PW` are the spatial padding sizes.
    - `OH` and `OW` are the the height and width of the output, respectively.
    """
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# =============================================================================
#  numpy im2col
# =============================================================================
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    img = np.pad(img,
                 ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                 mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
        
    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                   dtype=col.dtype)
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
    return img[:, :, PH:H + PH, PW:W + PW]

