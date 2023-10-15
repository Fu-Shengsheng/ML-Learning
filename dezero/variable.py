import numpy as np
from function import mul, add

class Variable:
    def __init__(self, data, name=None):
        # 当传入的数据不是 np.ndarray 类型时，提示类型不受支持的错误
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        # 为变量设置名称，默认为None
        self.name = name
        self.grad = None
        # 存放计算得到 variable 实例的函数
        self.creator = None
        # 初始化generation，标记变量在计算图中的代数（节点位置）
        self.generation = 0
    
    # 定义特殊方法 __len__，实现可以对 Variable 实例应用 len 函数
    # 如： x = Variable(np.array([1,2,3],[4,5,6])); len(x)
    def __len__(self):
        return len(self.data)
    
    # 重写 __repr__ 方法对 print 作用后的实例输出的字符串进行自定义
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        # str 将元素转为字符串类型
        # 换行后自动插入9个空格，对应 variable( 的长度，使得输出值对齐 
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    # 重写 __mul__ 实现乘法运算符 * 的重载
    # 此处为实例在 * 左侧时的重载计算
    def __mul__(self, other):
        return mul(self, other)
    
    def __add__(self, other):
        return add(self, other)
    
    # 使用 @property 装饰器，使得 shape 方法可以作为实例变量被访问
    # 如： x = Variable(np.array([1,2,3],[4,5,6])); 可以直接取 x.shape 而非 x.shape()
    @property
    def shape(self):
        return self.data.shape
    
    # 维度
    @property
    def ndim(self):
        return self.data.ndim
    
    # 元素数
    @property
    def size(self):
        return self.data.size
    
    # 数据类型
    @property
    def type(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1


    def backward(self, retain_grad=False):
        # # 获取创造函数
        # f = self.creator
        # if f is not None:
        #     # 创造函数输入变量的导数等于向创造函数传入当前值（输出值）的导数
        #     x = f.input
        #     x.grad = f.backward(self.grad)
        #     # 获取导数后进行后续的反向传播，以实现自动反向传播
        #     x.backward()

        # 为反向传播的起始节点（正向传播的结果）添加起始导数（即dy/dy=1）
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 存放调用函数
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                # 按代排序
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            # 获取函数，pop弹出数组最后一个元素
            f = funcs.pop()

            # 获取函数的输入输出
            # x, y = f.input, f.output
            # 计算导数
            # x.grad = f.backward(y.grad)
            
            # 将输出变量的导数存储在列表中
            # 由于 f.outputs 是基于 weakref 的弱引用，故此处需要使用 output() 获取引用的值
            gys = [output().grad for output in f.outputs]
            # 对输出列表进行解包并调用反向传播
            gxs = f.backward(*gys)
            # 当gxs不是元组时，将其转化为元组（比如加法运算返回的是元组，就不需要转换）
            # 以保障下一步遍历计算的通用性
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            # inputs 和 gxs 是一一对应的
            # 即参数f.inputs[i]的导数就是gxs[i]
            # 遍历zip(f.inputs, gxs)为每个输入参数x赋导数
            for x, gx in zip(f.inputs, gxs):
                # 针对性的处理同一个输入值在计算中重复使用的问题
                # 如果x已经赋予过导数，则新一次赋予时需要累加
                if x.grad == None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                # 如果当前反向传播的前一个环节仍然有函数处理，则循环执行
                # 实现自动反向传播
                if x.creator is not None:
                    add_func(x.creator)
            
            # 清除不需要保留的中间过程值的导数
            if not retain_grad:
                for y in f.outputs:
                    # y是weakref
                    y().grad = None

    # 清除梯度，每次进行新运算前调用，防止同一个变量在运算中的梯度累计了前次运算的结果
    def cleargrad(self):
        self.grad = None
