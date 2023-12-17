import contextlib
import numpy as np
import weakref
import dezero

# 控制是否启用反向传播
# 当进行推理验证阶段时，不需要启用，不保留计算之间的连接关系，以节省内存空间
class Config:
    enable_backprop = True

# 借助 @contextlib.contextmanager 装饰器创建一个判断上下文的函数
# 该函数搭配 with 使用，进入 with 时预处理被调用，离开 with 时后处理被调用
@contextlib.contextmanager
def using_config(name, value):
    # 预处理
    old_val = getattr(Config, name)
    setattr(Config, name, value)
    try:
        # 程序暂停
        yield
    finally:
        # 后处理
        setattr(Config, name, old_val)

def no_grad():
    return using_config('enable_backprop', False)

# np.ndarray 转换函数
def as_array(x):
    # 判断判断x是否为 ndarray 类型
    if np.isscalar(x):
        return np.array(x)
    return x

# 将传入的参数类型转为 Variable 实例
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

# Variable 变量类
class Variable:
    # 提升实例运算符的优先级，使其高于 ndarray 实例中重载的运算符
    __array_priority__ = 200
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
    def dtype(self):
        return self.data.dtype
    
    # 转置矩阵
    @property
    def T(self):
        return dezero.functions.transpose(self)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1


    def backward(self, retain_grad=False, create_graph=False):
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
            # self.grad = np.ones_like(self.data)
            # 将导数存储为 Variable 的实例，方便通过导数的反向传播求出二阶及高阶导数
            self.grad = Variable(np.ones_like(self.data))

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

            # 实现当 create_graph 为 false 时，反向传播的计算是在禁用反向传播模式下进行的
            with using_config('enable_backprop', create_graph): 
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

    # reshape 变形方法
    def reshape(self, *shape):
        # 实现对reshape([2,3]),reshape((2,3))形式调用的兼容
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    # 转置
    def transpose(self):
        return dezero.functions.transpose(self)
    
    # 求和
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

class Paramter(Variable):
    pass

# Function基类定义
class Function:
    # call 的声明使类的实例可以当做可调用对象使用
    # *inputs 表示所有的参数一次性拿到，不限参数数量
    def __call__(self, *inputs):
        # 将传入的参数类型转为 Variable
        inputs = [as_variable(x) for x in inputs]

        # 参数和返回值支持列表
        xs = [x.data for x in inputs]
        # 使用 * 号对 xs 进行解包
        ys = self.forward(*xs)

        # 对非元组的计算值额外处理，以保障下一步的续遍历计算的通用性
        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            # 为列表list的每个元素添加creator信息
            for output in outputs:
                # 输出变量保存创造者信息
                output.set_creator(self)

            # 保存输入值
            self.inputs = inputs
            # 保存输出值
            # 此处使用 weakref 创建outputs的弱引用，避免直接引用outputs导致的循环依赖
            self.outputs = [weakref.ref(output) for output in outputs]

        # 如果列表中只有一个元素，则返回第一个元素
        return outputs if len(outputs) > 1 else outputs[0]

    # 前向传播（计算结果）
    def forward(self, xs):
        # 抛出异常，表示这个方法应该通过继承实现
        raise NotImplementedError()

    # 反向传播（计算导数）
    def backward(self, gys):
        raise NotImplementedError()

# Square 类继承自 Function 类
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # x = self.input.data
        # 支持可变长输入&输出
        # x = self.inputs[0].data
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        # x = self.input.data
        # 支持可变长输入&输出
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
    
class Add(Function):
    def forward(self, x0, x1):
        # 对于不同形状的 ndarray，numpy 会在内部对计算进行自动进行广播计算，使两个加数的形状一致
        # 为了backward的顺利进行，需要存储两个加数的形状
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    # 加法运算的反向传播系数是1
    # 二元加法运算，返回一个 gy, gy 的元组
    def backward(self, gy):
        gx0, gx1 = gy, gy
        # 通过求梯度的和，使参数的梯度形状与参数保持一致
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Mul(Function):
    def forward(self, x0, x1):
        # 对于不同形状的 ndarray，numpy 会在内部对计算进行自动进行广播计算，使两个乘数的形状一致
        # 为了backward的顺利进行，需要存储两个乘数的形状
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data,  self.inputs[1].data
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0, gx1 = gy * x1, gy * x0
        # 通过求梯度的和，使参数的梯度形状与参数保持一致
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

# 负数运算（negative number arithmetic）
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
# 减法运算（Subtraction）
class Sub(Function):
    def forward(self, x0, x1):
        # 对于不同形状的 ndarray，numpy 会在内部对计算进行自动进行广播计算，使减数和被减数的形状一致
        # 为了backward的顺利进行，需要存储减数和被减数的形状
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, -gy
        # 通过求梯度的和，使参数的梯度形状与参数保持一致
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

# 除法运算（Division operation）
class Div(Function):
    def forward(self, x0, x1):
        # 对于不同形状的 ndarray，numpy 会在内部对计算进行自动进行广播计算，使被除数和除数的形状一致
        # 为了backward的顺利进行，需要存储被除数和除数的形状
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data,  self.inputs[1].data
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        # 通过求梯度的和，使参数的梯度形状与参数保持一致
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

# 求幂运算（exponentiation）
class Pow(Function):
    # c 为指数
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    
# 定义可供直接调用的函数
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def add(x0, x1):
    # 当 Variable 实例位于 + 号左侧时会调用，此时需要将 + 号右侧的数据也转为 ndarray 类型
    x1 = as_array(x1)
    f = Add()
    return f(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    f = Mul()
    return f(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    f = Sub()
    return f(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    f = Sub()
    # 交换减数和被减数
    return f(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x1, x0)

def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    # 重写 __add__ 实现乘法运算符 + 的重载
    # 此处为 Variable 实例在 + 左侧时的重载计算
    Variable.__add__ = add
    # 实现 Variable 实例在 * 右侧的重载
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezero.functions.get_item