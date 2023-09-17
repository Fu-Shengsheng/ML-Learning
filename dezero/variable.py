class Variable:
    def __init__(self, data):
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
        
        # 存放调用函数
        funcs = [self.creator]
        while funcs:
            # 获取函数，pop弹出数组最后一个元素
            f = funcs.pop()
            # 获取函数的输入输出
            x, y = f.input, f.output
            # 计算导数
            x.grad = f.backward(y.grad)

            # 如果当前反向传播的前一个环节仍然有函数处理，则循环执行
            # 实现自动反向传播
            if x.creator is not None:
                funcs.append(x.creator)