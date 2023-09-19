import numpy as np


class Variable:
    def __init__(self, data):
        # 当传入的数据不是 np.ndarray 类型时，提示类型不受支持的错误
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        # 存放计算得到 variable 实例的函数
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
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
        funcs = [self.creator]
        while funcs:
            # 获取函数，pop弹出数组最后一个元素
            f = funcs.pop()

            # 获取函数的输入输出
            # x, y = f.input, f.output
            # 计算导数
            # x.grad = f.backward(y.grad)
            
            # 将输出变量的导数存储在列表中
            gys = [output.grad for output in f.outputs]
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
                x.grad = gx

                # 如果当前反向传播的前一个环节仍然有函数处理，则循环执行
                # 实现自动反向传播
                if x.creator is not None:
                    funcs.append(x.creator)
