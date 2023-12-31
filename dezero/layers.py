import os
import numpy as np
import weakref
import dezero.functions as F
from dezero.core import Paramter
from dezero.utils import pair

class Layer:
    def __init__(self):
        self._params = set()

    # __setattr__ 用于在设置对象属性时调用
    def __setattr__(self, name, value):
        # Layer 支持持有 Parameter 和其他 Layer 实例
        if isinstance(value, (Paramter, Layer)):
            self._params.add(name)

        # 为实例挂载属性，此处super调用的父类是基类 Object
        # 可以通过 instance.__dict__[name] 访问指定属性
        super().__setattr__(name, value)

    # *inputs 表示接受任意数目的元素并将其打包成一个元组 tuple
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        
        # 实例通过弱引用持有输入和输出变量，方便gc，避免内存泄漏
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            # 从 Layer 实例中取出参数
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                # yield 暂停处理并返回值
                # 实现按顺序返回参数
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    # 递归取出 layer 中的所有参数，子层级下的参数为 layer/key，并挂载到 params_dict 对象
    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    # 保存参数
    def save_weights(self, path):
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    # 加载参数
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Paramter(None, name='W')

        # 如果没有指定 in_size, 则延后进行 W 参数初始化
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Paramter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        # 将 W 参数进行 * np.sqrt(1 / I) 运算后作为初始化是一种常用策略
        # 称为 "Xavier" 或 "Glorot" 初始化
        # 目标是在不同层之间保持输入和输出的方差相对一致，以促使信息在网络中更好地流动
        # 乘以 np.sqrt(1 / I) 可以将权重矩阵的方差缩小，这有助于防止梯度消失或梯度爆炸问题
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
        
    def forward(self, x):
        # 在传播数据时根据传入的 x 的形状初始化权重
        if self.W.data is None:
            # x.shape[1] 表示取每行的元素数目，即输入值
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Paramter(None, name='W')

        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Paramter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        # 初始化权重的缩放因子
        scale = np.sqrt(1 / (C * KH * KW))
        # 生成的数据乘以 scale 作为权重是为了防止 W 的值过大或者过小，出现梯度消失或者爆炸的情况
        W_data = np.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W()
        
        y = F.conv2d_simple(x, self.W, self.b, self.stride, self.pad)
        return y
    

class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        # x2h 全连接层声明，初始化参数
        self.x2h = Linear(hidden_size, in_size=in_size)
        # h2h 全连接层声明，初始化参数
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        # 对于首次计算，对原始输入值进行 x2h 计算后执行激活
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        # 后续每次计算都基于之前得到的 h 进行 h2h 计算，并加上本次的 x2h 计算
        # 这样就建立了本轮运算的数据与前一轮运算的数据的关联关系
        else:
            # 对于同一个 RNN Layer，因为 hidden_size（即输出形状） 已经确定，故前后两轮形状相同，可以相加
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        
        self.h = h_new
        return h_new